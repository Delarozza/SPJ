import re

print(re.findall(r"(ab)+", "ababab"))
print(re.findall(r"\d{3}", "1 12 123 1234 12345"))  # ['12', '123', '1234', '1234']