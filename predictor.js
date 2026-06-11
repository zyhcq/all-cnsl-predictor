// ==========================================
// predictor.js - Domain Concept Cox Model Engine
// ==========================================

// Global radar chart instance
let radarChartInstance = null;

// Removed toggle state for Option B

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
        age_months: 130, sex: 1, risk_code: 3, immuno_clean: "T", fusion_gene_code: 0, karyo_abn: 1,
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
    // Validation for mandatory fields
    const ageRaw = document.getElementById('age_months').value;
    const sexRaw = document.getElementById('sex').value;
    const riskRaw = document.getElementById('risk_code').value;

    if (ageRaw === "" || sexRaw === "" || riskRaw === "") {
        alert("Please provide the mandatory Baseline Demographics (Age, Sex, and Clinical Risk Stratification) before calculating.");
        return;
    }

    // 1. Gather Inputs & Impute
    let d = {};
    const features = [
        'age_months', 'wbc', 'anc', 'monocyte', 'plt', 'ldh', 'uric_acid', 'ast', 'alt', 'albumin', 'alp',
        'csf_wbc', 'csf_protein', 'csf_glucose', 'crp', 'pt', 'aptt', 'fibrinogen', 'd_dimer', 'alc',
        'pct', 'prealbumin', 'csf_rbc'
    ];
    
    let missingCount = 0;

    features.forEach(f => {
        let val = getVal(f);
        if (val === null) {
            val = MODEL_CONFIG.impute_params[f] || 0;
            if (f !== 'age_months') missingCount++;
        }
        d[f] = val;
    });

    if (document.getElementById('immuno_clean').value === "") missingCount++;
    if (document.getElementById('fusion_gene_code').value === "") missingCount++;
    if (document.getElementById('karyo_abn').value === "") missingCount++;

    const warningEl = document.getElementById('missing-data-warning');
    if (warningEl) {
        if (missingCount > 7) {
            warningEl.style.display = 'block';
        } else {
            warningEl.style.display = 'none';
        }
    }

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

    let karyo = getVal('karyo_abn') || 0;
    z.karyo_abn = karyo === 1 ? 1 : 0;

    // 6. SPCA Transformation
    let concepts = {};
    for (const [cName, pData] of Object.entries(MODEL_CONFIG.spca_params)) {
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

    // 8. Linear Predictors (LP)
    let lp_r = 0;
    let lp_d = 0;
    const coef_r = MODEL_CONFIG.cox_relapse.coefficients;
    const coef_d = MODEL_CONFIG.cox_death.coefficients;
    const mean_r = MODEL_CONFIG.cox_relapse.norm_mean || {};
    const mean_d = MODEL_CONFIG.cox_death.norm_mean || {};
    
    const features_to_add = [
        {name: 'C1_TumorBurden_SPCA', val: concepts['C1_TumorBurden']},
        {name: 'C2_ImmuneProfile_SPCA', val: concepts['C2_ImmuneProfile']},
        {name: 'C3_CNSInvasion_SPCA', val: concepts['C3_CNSInvasion']},
        {name: 'C4_GeneticRisk_SPCA', val: concepts['C4_GeneticRisk']},
        {name: 'C5_InflamCoag_SPCA', val: concepts['C5_InflamCoag']},
        {name: 'C6_NutriLiver_SPCA', val: concepts['C6_NutriLiver']},
        {name: 'is_high_risk', val: is_high_risk},
        {name: 'is_high_risk_age', val: is_high_risk_age},
        {name: 'is_male', val: is_male}
    ];

    features_to_add.forEach(f => {
        let centered_r = f.val - (mean_r[f.name] || 0);
        let centered_d = f.val - (mean_d[f.name] || 0);
        lp_r += centered_r * (coef_r[f.name] || 0);
        lp_d += centered_d * (coef_d[f.name] || 0);
    });

    let hr_relapse = Math.exp(lp_r);
    let hr_death = Math.exp(lp_d);

    // 9. Cumulative Incidence Function (CIF) for Competing Risks
    function getH0(times, hazards, t) {
        let h = 0;
        for (let i = 0; i < times.length; i++) {
            if (times[i] <= t) {
                h = hazards[i];
            } else {
                break;
            }
        }
        return h;
    }

    let times_r = MODEL_CONFIG.cox_relapse.baseline_cumulative_hazard.times;
    let times_d = MODEL_CONFIG.cox_death.baseline_cumulative_hazard.times;
    let all_times = Array.from(new Set([...times_r, ...times_d])).sort((a, b) => a - b);
    
    let cif_r_3yr = 0;
    let cif_r_5yr = 0;
    let cif_d_3yr = 0;
    let cif_d_5yr = 0;
    let prev_S = 1.0;
    let prev_H_R = 0;
    let prev_H_D = 0;
    
    for (let i = 0; i < all_times.length; i++) {
        let t = all_times[i];
        
        let h0_r = getH0(times_r, MODEL_CONFIG.cox_relapse.baseline_cumulative_hazard.hazards, t);
        let h0_d = getH0(times_d, MODEL_CONFIG.cox_death.baseline_cumulative_hazard.hazards, t);
        
        let H_R = h0_r * hr_relapse;
        let H_D = h0_d * hr_death;
        
        let dH_R = H_R - prev_H_R;
        let dH_D = H_D - prev_H_D;
        
        // Aalen-Johansen discrete approximation
        if (t <= 1095) {
            cif_r_3yr += prev_S * (1 - Math.exp(-dH_R));
            cif_d_3yr += prev_S * (1 - Math.exp(-dH_D));
        }
        if (t <= 1825) {
            cif_r_5yr += prev_S * (1 - Math.exp(-dH_R));
            cif_d_5yr += prev_S * (1 - Math.exp(-dH_D));
        }
        
        prev_S = Math.exp(-(H_R + H_D));
        prev_H_R = H_R;
        prev_H_D = H_D;
    }

    let risk_3yr = cif_r_3yr * 100;
    let risk_5yr = cif_r_5yr * 100;
    let risk_d_3yr = cif_d_3yr * 100;
    let risk_d_5yr = cif_d_5yr * 100;

    // 10. Update UI
    document.getElementById('risk-3yr').innerText = risk_3yr.toFixed(2) + "%";
    document.getElementById('risk-5yr').innerText = risk_5yr.toFixed(2) + "%";
    document.getElementById('risk-death-3yr').innerText = risk_d_3yr.toFixed(2) + "%";
    document.getElementById('risk-death-5yr').innerText = risk_d_5yr.toFixed(2) + "%";
    
    // Progress bar fill (max at 50% risk for full visual impact)
    document.getElementById('risk-bar-3yr').style.width = Math.min((risk_3yr / 50.0) * 100, 100) + "%";
    document.getElementById('risk-bar-5yr').style.width = Math.min((risk_5yr / 50.0) * 100, 100) + "%";
    document.getElementById('risk-bar-death-3yr').style.width = Math.min((risk_d_3yr / 50.0) * 100, 100) + "%";
    document.getElementById('risk-bar-death-5yr').style.width = Math.min((risk_d_5yr / 50.0) * 100, 100) + "%";

    // 11. Calculate Contributions for Explainable AI
    const contribsR = [
        { name: 'C1. Tumor Burden', val: (concepts['C1_TumorBurden'] - (mean_r['C1_TumorBurden_SPCA']||0)) * (coef_r['C1_TumorBurden_SPCA'] || 0) },
        { name: 'C2. Immune Profile', val: (concepts['C2_ImmuneProfile'] - (mean_r['C2_ImmuneProfile_SPCA']||0)) * (coef_r['C2_ImmuneProfile_SPCA'] || 0) },
        { name: 'C3. CNS Invasion', val: (concepts['C3_CNSInvasion'] - (mean_r['C3_CNSInvasion_SPCA']||0)) * (coef_r['C3_CNSInvasion_SPCA'] || 0) },
        { name: 'C4. Genetic Risk', val: (concepts['C4_GeneticRisk'] - (mean_r['C4_GeneticRisk_SPCA']||0)) * (coef_r['C4_GeneticRisk_SPCA'] || 0) },
        { name: 'C5. Inflam & Coag', val: (concepts['C5_InflamCoag'] - (mean_r['C5_InflamCoag_SPCA']||0)) * (coef_r['C5_InflamCoag_SPCA'] || 0) },
        { name: 'C6. Nutri & Liver', val: (concepts['C6_NutriLiver'] - (mean_r['C6_NutriLiver_SPCA']||0)) * (coef_r['C6_NutriLiver_SPCA'] || 0) },
        { name: 'High Risk Stratification', val: (is_high_risk - (mean_r['is_high_risk']||0)) * (coef_r['is_high_risk'] || 0) },
        { name: 'Age Risk', val: (is_high_risk_age - (mean_r['is_high_risk_age']||0)) * (coef_r['is_high_risk_age'] || 0) },
        { name: 'Sex (Male)', val: (is_male - (mean_r['is_male']||0)) * (coef_r['is_male'] || 0) }
    ];

    const contribsD = [
        { name: 'C1. Tumor Burden', val: (concepts['C1_TumorBurden'] - (mean_d['C1_TumorBurden_SPCA']||0)) * (coef_d['C1_TumorBurden_SPCA'] || 0) },
        { name: 'C2. Immune Profile', val: (concepts['C2_ImmuneProfile'] - (mean_d['C2_ImmuneProfile_SPCA']||0)) * (coef_d['C2_ImmuneProfile_SPCA'] || 0) },
        { name: 'C3. CNS Invasion', val: (concepts['C3_CNSInvasion'] - (mean_d['C3_CNSInvasion_SPCA']||0)) * (coef_d['C3_CNSInvasion_SPCA'] || 0) },
        { name: 'C4. Genetic Risk', val: (concepts['C4_GeneticRisk'] - (mean_d['C4_GeneticRisk_SPCA']||0)) * (coef_d['C4_GeneticRisk_SPCA'] || 0) },
        { name: 'C5. Inflam & Coag', val: (concepts['C5_InflamCoag'] - (mean_d['C5_InflamCoag_SPCA']||0)) * (coef_d['C5_InflamCoag_SPCA'] || 0) },
        { name: 'C6. Nutri & Liver', val: (concepts['C6_NutriLiver'] - (mean_d['C6_NutriLiver_SPCA']||0)) * (coef_d['C6_NutriLiver_SPCA'] || 0) },
        { name: 'High Risk Stratification', val: (is_high_risk - (mean_d['is_high_risk']||0)) * (coef_d['is_high_risk'] || 0) },
        { name: 'Age Risk', val: (is_high_risk_age - (mean_d['is_high_risk_age']||0)) * (coef_d['is_high_risk_age'] || 0) },
        { name: 'Sex (Male)', val: (is_male - (mean_d['is_male']||0)) * (coef_d['is_male'] || 0) }
    ];
    
    document.getElementById('contributors-container-r').style.display = 'block';
    document.getElementById('contributors-container-d').style.display = 'block';
    
    renderPanel('contributors-list-r', contribsR, '#e0f2fe', '#0ea5e9');
    renderPanel('contributors-list-d', contribsD, '#fee2e2', '#ef4444');

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

function initRadarChart(data) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Tumor Burden', 'Immune Profile', 'CNS Invasion', 'Genetic Risk', 'Inflammation & Coag', 'Nutri & Liver'],
            datasets: [{
                label: 'Concept Score (Z-score)',
                data: data,
                backgroundColor: 'rgba(14, 165, 233, 0.2)',
                borderColor: '#0ea5e9',
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#0284c7',
                pointHoverBackgroundColor: '#ffffff',
                pointHoverBorderColor: '#0ea5e9',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(0, 0, 0, 0.1)' },
                    grid: { color: 'rgba(0, 0, 0, 0.1)' },
                    pointLabels: {
                        color: '#475569',
                        font: { size: 12, family: 'Inter', weight: '500' }
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

function renderPanel(listId, contribs, bgWrapper, barColor) {
    let riskDrivers = contribs.filter(c => c.val > 0).sort((a, b) => b.val - a.val);
    const listEl = document.getElementById(listId);
    
    if (riskDrivers.length > 0) {
        listEl.style.display = 'block';
        listEl.innerHTML = '';
        const maxVal = riskDrivers[0].val;
        const topDrivers = riskDrivers.slice(0, 4); 
        
        topDrivers.forEach(c => {
            const pct = Math.max((c.val / maxVal) * 100, 5); 
            listEl.innerHTML += `
                <div class="contributor-item">
                    <div class="contributor-name">${c.name}</div>
                    <div class="contributor-bar-wrapper" style="background: ${bgWrapper}">
                        <div class="contributor-bar-fill" style="width: ${pct}%; background: ${barColor}"></div>
                    </div>
                    <div class="contributor-value" style="color: ${barColor}">+${c.val.toFixed(2)}</div>
                </div>
            `;
        });
    } else {
        listEl.innerHTML = '<div style="color: #64748b; font-size: 0.9rem; padding: 1rem 0;">No significant positive risk drivers found for this outcome.</div>';
    }
}
