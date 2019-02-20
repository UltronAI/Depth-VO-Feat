f_l1 = open('train_left_1.txt', 'r')
f_l2 = open('train_left_2.txt', 'r')
f_r1 = open('train_right_1.txt', 'r')
f_r2 = open('train_right_2.txt', 'r')
f_k = open('train_K.txt', 'r')
f_t = open('train_T_R2L.txt', 'r')

f_out = open('train.txt', 'w')

lines_l1 = f_l1.readlines()
lines_l2 = f_l2.readlines()
lines_r1 = f_r1.readlines()
lines_r2 = f_r2.readlines()
lines_k = f_k.readlines()
lines_t = f_t.readlines()

for i in range(len(lines_t)):
    line_l1 = lines_l1[i].strip()
    line_l2 = lines_l2[i].strip()
    line_r1 = lines_r1[i].strip()
    line_r2 = lines_r2[i].strip()
    line_k = lines_k[i].strip()
    line_t = lines_t[i].strip()

    new_line = " ".join([line_l1, line_l2, line_r1, line_r2, line_k, line_t])
    f_out.writelines(new_line + '\n')

f_l1.close()
f_l2.close()
f_r1.close()
f_r2.close()
f_k.close()
f_t.close()
f_out.close()