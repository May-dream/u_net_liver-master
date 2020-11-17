'''
   黄哥Python 联系方式，自己搜
'''


class STgetitem:

    def __init__(self, text):
        self.text = text

    def __getitem__(self, index):
        result = self.text[index].upper()
        return result


p = STgetitem("黄哥Python")
print(p[0])
print("------------------------")
for char in p:
    print(char)