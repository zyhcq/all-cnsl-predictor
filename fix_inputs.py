import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace <input type="number" ... > to include min="0"
new_content = re.sub(r'(<input[^>]*type=["\']number["\'])(?!.*?min=)([^>]*)>', r'\1 min="0"\2>', content)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Updated index.html inputs.')
