from pathlib import Path

source = Path("data/raw/asts.csv")
target = Path("data/asts_fixed.csv")

text = source.read_text(encoding="utf-8-sig")
import re

pattern = r'""([^"\r\n]*,[^"\r\n]*)""'
matches = re.findall(pattern, text)

print("找到的异常字段数：", len(matches))
for value in matches:
    print(repr(value))
fixed, count = re.subn(
    pattern,
    lambda match: '"' + match.group(1) + '"',
    text
)
target.write_text(fixed, encoding="utf-8")

print(f"修复了 {count} 处")
print(f"新文件：{target}")
