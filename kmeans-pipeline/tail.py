import operator
import sys
import csv

alpha = .1

f1 = open(sys.argv[1])
f2 = open(sys.argv[1])
f3 = open(sys.argv[1])

mydict = {}
final_count = {}
for line in f1: 
    #`print("asasasasas")
    words = line.split()
    for word in words: 
        if word not in mydict:
            mydict[word] = 0
            final_count[word] = 0  
for i, line in enumerate(f2):
    #print(j) 
    words = line.split()
    #print("asas")
    for w in words:
      # print(w)
      mydict[w] = mydict[w] + 1		

word_dict_list = sorted(mydict.items(), key=operator.itemgetter(1),reverse=True)
# print(word_dict_list)
# k  = dict(word_dict_list)

w = csv.writer(open("sorted_word_dist_2.txt", "w"))
for key, val in word_dict_list:
	w.writerow([key, val])
#print("assas")

m = (1.0*word_dict_list[0][1])/len(word_dict_list)
# print(m)
word_dict_list_copy = word_dict_list
slope = alpha*m
sampled_disc = []
n_samples = int(sys.argv[2])

indicing_dict = {}
for i in range(len(word_dict_list)):
	indicing_dict[word_dict_list[i][0]] = i
#print("Ass")
line_file = open("lines_sampled",'wb')
i = 0
j = 0
for line in f3:
  if(j%1000==0):
    print(j)
  j = j + 1
  if(i== n_samples):
    break
  words = line.split()
  flag = 0
  for word in words:
   #ADD THE IF CONDITION!!!!!!!!!! FOR FINISHING 	
   if ((len(word_dict_list)-indicing_dict[word])*slope > final_count[word]):
     final_count[word] = final_count[word] + 1
     continue
   else:
  		# print("yabba wabba dubb dubb")
  		flag = 1
  if(flag == 0):
  	line_file.write(str(i)+'\n')
  	sampled_disc.append(line)
  	i = i + 1


print("asdadadadd  " + str(i))
thefile = open(sys.argv[1], 'w')
for item in sampled_disc:
  thefile.write("%s" % item)
