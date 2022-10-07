


num_atr=[10,8,70,16,7,14,6,5,2,100,40,100,40]
fixed = [0,0,0,0,0,0,0,0,1,0,0,0,0]
fix_atr = []
num=1
for i in range(0,len(fixed)):
    if(fixed[i]==1):
        num = num*num_atr[i]
        fix_atr.append(i)
print(fix_atr, num)

max = -1
min = 100
# print num
val = 0

while val < num:
    inp_fix = ['', '', '', '', '', '', '', '', '', '', '', '', '']
    i = len(fix_atr) - 1
    tmp_val = val
    # if(val%10000==0):
    # print val
    while i >= 0:
        inp_fix[fix_atr[i]] = tmp_val % num_atr[fix_atr[i]]
        tmp_val = (tmp_val - tmp_val % num_atr[fix_atr[i]]) / num_atr[fix_atr[i]]

        print(inp_fix[fix_atr[i]], tmp_val)
        i -= 1
    # print inp_fix
    val += 1

print(inp_fix)

