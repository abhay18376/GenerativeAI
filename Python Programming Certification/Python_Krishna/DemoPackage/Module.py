import re
def multi_Split(filename):
    fp=open(filename,'r')
    reg_object=re.compile('[@;-]')
    result_out=[]
    for each_line in fp:
        each_line=each_line.strip()
        result=reg_object.split(each_line)
        result_out.append(result)
    return result_out