from sys import argv
import os
import re

#test_char="The cost time :47(us), size: 1"
#print("test char:", test_char)

def para_cost_time(exec_result):
    first_result = re.search('^The cost time :[0-9]+\(us\)',exec_result,re.S)
    if None == first_result:
        return None
    #print(first_result.group(0))
    cost_result = re.search('[0-9]+',first_result.group(0),re.S)
    #print(cost_result.group(0))
    return cost_result.group(0)

def para_buffer_size(exec_result):
    check_head = re.search('^The cost time',exec_result,re.S)
    if None == check_head:
        return None
    first_result = re.search('size: [0-9]+',exec_result,re.S)
    if None == first_result:
        return None
    #print(first_result.group(0))
    buffer_size = re.search('[0-9]+',first_result.group(0),re.S)
    #print(buffer_size.group(0))
    return buffer_size.group(0)
    
def para_result(target):    
    #target = open(filename)
    line = target.readline()
    #para_buffer_size(test_char)
    while line:
        print(line)         
        cost_time = para_cost_time(line)
        if (None != cost_time):
            buffer_size = para_buffer_size(line)
            #print("cost_time: ",cost_time,"size: ",buffer_size)
            if (None != buffer_size):
                if (buffer_size in size_index_list):
                    index = size_index_list.index(buffer_size)
                    number_index_list[index] +=1
                    time_index_list[index] += int(cost_time)
                else:
                    size_index_list.append(buffer_size)
                    number_index_list.append(1)
                    result = int(cost_time)
                    time_index_list.append(result)                    
        line = target.readline()
        
    #target.close()
def do_result():
    for index in range(len(time_index_list)):
        print("size: %-10s cost time:%10.2f, repeat run count:%-10d"%(size_index_list[index], (time_index_list[index]/number_index_list[index]),number_index_list[index]))
    pass  
        
def main():
    commandstr = hiptest + " " + testsuite
    #print("command:", commandstr)
    #cmdout=os.popen(commandstr);
    case_num = 1;
    while case_num < 32:
        commandstr = hiptest + " " +"--gtest_filter="+testsuite+"."+testsuite+"_"+ str(case_num)
        #print("command:", commandstr)
        repeat_num = 0
        while repeat_num < 10:
            cmdout=os.popen(commandstr)
            para_result(cmdout)
            repeat_num += 1    
        case_num +=1
    #line = cmdout.readline()
    #para_result(cmdout);
    do_result()
    pass 

script,hiptest,testsuite = argv
print("hiptest:", hiptest)
print("testsuite:", testsuite)
size_index_list=[]
number_index_list=[]
time_index_list=[]
main()
