from sys import argv
import os
import re


global size_index_list
global number_index_list
global time_index_list
size_index_list=[]
number_index_list=[]
time_index_list=[]

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
def do_result_w():
    doc=open('out.txt','w')
    print("\n",file=doc,end=' ')
    print("size     ",file=doc,end=' ')
    for index in range(len(time_index_list)):
        print(" %-10s"%(size_index_list[index]),file=doc,end=' ')
    #print("\n",file=doc)
    doc.close()
    doc=open('out.txt','a')
    print("\n",file=doc,end=' ')
    print(testsuite[0],file=doc,end=' ')
    for index in range(len(time_index_list)):
        print(" %10.2f"%((time_index_list[index]/number_index_list[index])),file=doc,end=' ')
    doc.close()
    pass  

def do_result_a():
    doc=open('out.txt','a')
    #print("\n",file=doc,end=' ')
    #print("cost_time",file=doc,end=' ')
    #print("\n",file=doc)
    for index in range(len(time_index_list)):
        print(" %10.2f"%((time_index_list[index]/number_index_list[index])),file=doc,end=' ')
    doc.close()
    pass  
        
def main():
    Num_test=0
    while Num_test < len(testsuite) :
        commandstr = hiptest + " " + testsuite[Num_test]
        print("command:", commandstr)
        #cmdout=os.popen(commandstr);
        case_num = 1;
        while case_num < 32:
            commandstr = hiptest + " " +"--gtest_filter="+testsuite[Num_test]+"."+testsuite[Num_test]+"_"+ str(case_num)
            print("command:", commandstr)
            repeat_num = 0
            while repeat_num < 10:
                cmdout=os.popen(commandstr)
                para_result(cmdout)
                repeat_num += 1    
            case_num +=1
        if Num_test !=0:
            doc=open('out.txt','a')
            print("\n",file=doc,end=' ')
            print(testsuite[Num_test],file=doc,end=' ')
            doc.close()
            do_result_a()
            #print(size_index_list)
            #print(number_index_list)
            #print(time_index_list)
            del size_index_list[:]
            del number_index_list[:]
            del time_index_list[:]
        else:
            do_result_w()
            #print(size_index_list)
            #print(number_index_list)
            #print(time_index_list)
            del size_index_list[:]
            del number_index_list[:]
            del time_index_list[:]

        Num_test+=1
    pass 



testsuite=['hipPerformancehipMemset','hipPerformancepinH2D','hipPerformanceUnpinD2H','hipPerformanceUnpinH2D']
#testsuite=['hipPerformanceUnpinD2H','hipPerformanceUnpinH2D']
script,hiptest = argv
print("hiptest:", hiptest)
print("length of testsuit:", len(testsuite))
#print("testsuite:", testsuite[0])
main()
#print(size_index_list)
#print(number_index_list)
#print(time_index_list)