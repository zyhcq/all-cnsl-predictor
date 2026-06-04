// ==========================================
// predictor.js - Domain Concept Cox Model Engine
// ==========================================

// Global radar chart instance
let radarChartInstance = null;

// Ensure model config is loaded
if (typeof MODEL_CONFIG === 'undefined') {
    alert("Error: model_config.js failed to load!");
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('predict-btn').addEventListener('click', runPrediction);
    document.getElementById('fill-mock-btn').addEventListener('click', fillMockData);
    
    // Initialize empty chart
    initRadarChart([0,0,0,0,0,0]);
});

function getVal(id) {
    const v = document.getElementById(id).value;
    return v === "" ? null : Number(v);
}

function fillMockData() {
    // Fill with high-risk mock values
    const mock = {
        age_months: 130, sex: 1, risk_code: 3, immuno_clean: "T", fusion_gene_code: 0,
        wbc: 120, plt: 20, anc: 0.5, monocyte: 0.8, alc: 0.2,
        csf_wbc: 15, csf_rbc: 0, csf_protein: 0.6, csf_glucose: 2.1,
        ldh: 1500, uric_acid: 600, crp: 45, pct: 2.1,
        albumin: 30, prealbumin: 120, ast: 150, alt: 140, alp: 300,
        d_dimer: 5.5, pt: 15.2, aptt: 45, fibrinogen: 1.2
    };
    for (const [k, v] of Object.entries(mock)) {
        const el = document.getElementById(k);
        if (el) el.value = v;
    }
}

function runPrediction() {
    // 1. Gather Inputs & Impute
    let d = {};
    const features = [
        'age_months', 'wbc', 'anc', 'monocyte', 'plt', 'ldh', 'uric_acid', 'ast', 'alt', 'albumin', 'alp',
        'csf_wbc', 'csf_protein', 'csf_glucose', 'crp', 'pt', 'aptt', 'fibrinogen', 'd_dimer', 'alc',
        'pct', 'prealbumin', 'csf_rbc'
    ];
    
    features.forEach(f => {
        let val = getVal(f);
        if (val === null) {
            val = MODEL_CONFIG.impute_params[f] || 0;
        }
        d[f] = val;
    });

    // 2. Ratios & Safe Division
    let alc = d.alc === 0 ? 0.01 : d.alc;
    let wbc = d.wbc === 0 ? 0.01 : d.wbc;
    let alt = d.alt === 0 ? 0.01 : d.alt;
    let csf_prot = d.csf_protein === 0 ? 0.01 : d.csf_protein;

    d.nlr_raw = d.anc / alc;
    d.plr_raw = d.plt / alc;
    d.mlr_raw = d.monocyte / alc;
    d.sii_raw = (d.plt * d.anc) / alc;
    d.ast_alt_raw = d.ast / alt;
    d.lwr_raw = d.ldh / wbc;
    d.pni_raw = d.albumin + 5 * alc;
    d.csf_wbc_prot_raw = d.csf_wbc / csf_prot;

    // 3. Log Transforms
    const logCols = [
        'wbc', 'anc', 'monocyte', 'plt', 'ldh', 'uric_acid', 'ast', 'alt', 'alp',
        'csf_wbc', 'csf_protein', 'crp', 'aptt', 'd_dimer', 'pct', 'csf_rbc'
    ];
    logCols.forEach(c => d[c + '_log'] = Math.log1p(Math.max(d[c], 0)));

    const ratioCols = ['nlr', 'plr', 'mlr', 'sii', 'ast_alt', 'lwr', 'pni', 'csf_wbc_prot'];
    ratioCols.forEach(c => d[c + '_log'] = Math.log1p(Math.max(d[c + '_raw'], 0)));

    // 4. Z-Scores
    let z = {};
    const getZ = (colName, actualCol) => {
        const p = MODEL_CONFIG.scaler_params[colName];
        return p ? (d[actualCol] - p.mean) / p.std : 0;
    };

    logCols.forEach(c => z[c + '_log_z'] = getZ(c + '_log', c + '_log'));
    ratioCols.forEach(c => z[c + '_z'] = getZ(c + '_log', c + '_log')); // standard scaler used log
    const zOnly = ['age_months', 'albumin', 'csf_glucose', 'pt', 'fibrinogen', 'prealbumin'];
    zOnly.forEach(c => z[c + '_z'] = getZ(c, c));

    // 5. Categorical / Dummy
    z.wbc_high = d.wbc >= 50.0 ? 1 : 0;
    z.csf_wbc_positive = d.csf_wbc > 0 ? 1 : 0;
    z.csf_rbc_positive = d.csf_rbc > 0 ? 1 : 0;
    
    let immuno = document.getElementById('immuno_clean').value;
    z.is_T_ALL = (immuno === 'T') ? 1 : 0;

    let fusion = getVal('fusion_gene_code') || 0;
    z.fusion_TEL = fusion === 1 ? 1 : 0;
    z.fusion_E2A = fusion === 2 ? 1 : 0;
    z.fusion_BCR = fusion === 3 ? 1 : 0;
    z.fusion_MLL = fusion === 4 ? 1 : 0;
    z.fusion_other = [5,6,7].includes(fusion) ? 1 : 0;

    // 6. PCA Transformation
    let concepts = {};
    for (const [cName, pData] of Object.entries(MODEL_CONFIG.pca_params)) {
        let score = 0;
        for (let i = 0; i < pData.features.length; i++) {
            let f = pData.features[i];
            let w = pData.weights[i];
            let mean = pData.mean[i];
            score += ((z[f] || 0) - mean) * w;
        }
        concepts[cName] = score;
    }

    // 7. Cox Model Direct Features
    let age = d.age_months;
    let risk = getVal('risk_code') || 1;
    let sex = getVal('sex') || 2;
    
    let is_high_risk_age = (age < 12 || age >= 120) ? 1 : 0;
    let is_high_risk = (risk >= 2) ? 1 : 0;
    let is_male = (sex === 1) ? 1 : 0;

    // 8. Linear Predictor (LP)
    let lp = 0;
    const coef = MODEL_CONFIG.cox_params.coefficients;
    
    lp += concepts['C1_TumorBurden'] * (coef['C1_TumorBurden'] || 0);
    lp += concepts['C2_ImmuneProfile'] * (coef['C2_ImmuneProfile'] || 0);
    lp += concepts['C3_CNSInvasion'] * (coef['C3_CNSInvasion'] || 0);
    lp += concepts['C4_GeneticRisk'] * (coef['C4_GeneticRisk'] || 0);
    lp += concepts['C5_InflamCoag'] * (coef['C5_InflamCoag'] || 0);
    lp += concepts['C6_NutriLiver'] * (coef['C6_NutriLiver'] || 0);
    
    lp += is_high_risk * (coef['is_high_risk'] || 0);
    lp += is_high_risk_age * (coef['is_high_risk_age'] || 0);
    lp += is_male * (coef['is_male'] || 0);

    let hazardRatio = Math.exp(lp);

    // 9. Survival Probabilities
    let s0_3yr = getBaselineS0(36); // 36 months
    let s0_5yr = getBaselineS0(60); // 60 months

    let s_3yr = Math.pow(s0_3yr, hazardRatio);
    let s_5yr = Math.pow(s0_5yr, hazardRatio);

    let risk_3yr = (1 - s_3yr) * 100;
    let risk_5yr = (1 - s_5yr) * 100;

    // 10. Update UI
    document.getElementById('risk-3yr').innerText = risk_3yr.toFixed(2) + "%";
    document.getElementById('risk-5yr').innerText = risk_5yr.toFixed(2) + "%";
    
    // Progress bar fill (max at 50% risk for full visual impact)
    let fillPct = Math.min((risk_5yr / 50.0) * 100, 100);
    document.getElementById('risk-bar').style.width = fillPct + "%";

    // Update Radar Chart
    updateRadarChart([
        concepts['C1_TumorBurden'],
        concepts['C2_ImmuneProfile'],
        concepts['C3_CNSInvasion'],
        concepts['C4_GeneticRisk'],
        concepts['C5_InflamCoag'],
        concepts['C6_NutriLiver']
    ]);
}

function getBaselineS0(targetTime) {
    const times = MODEL_CONFIG.cox_params.baseline_survival.times;
    const probs = MODEL_CONFIG.cox_params.baseline_survival.probabilities;
    
    // Find closest time point (step function)
    let bestProb = 1.0;
    for (let i = 0; i < times.length; i++) {
        if (times[i] <= targetTime) {
            bestProb = probs[i];
        } else {
            break;
        }
    }
    return bestProb;
}

function initRadarChart(data) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['肿瘤负荷', '免疫微环境', '中枢侵袭', '遗传风险', '炎症与凝血', '营养与肝功能'],
            datasets: [{
                label: '概念特征得分 (Z-score)',
                data: data,
                backgroundColor: 'rgba(14, 165, 233, 0.2)',
                borderColor: '#38bdf8',
                pointBackgroundColor: '#f8fafc',
                pointBorderColor: '#0284c7',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#38bdf8',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' }
                    },
                    ticks: {
                        display: false, // hide numbers on axis
                        min: -5,
                        max: 5
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateRadarChart(data) {
    if (radarChartInstance) {
        radarChartInstance.data.datasets[0].data = data;
        radarChartInstance.update();
    }
}
