from __future__ import print_function
import csv
import sys
import itertools
import collections
import sys
import re

csv.field_size_limit(sys.maxsize)

mydict2={}

def replace_to_single():
	mydict={}
	for key2,value in mydict2.items():
		ch=[',',':','[',']','-','{','}','(',')',';','+',"'",'~','"','!']
		ch2=['{','}','[',']',';']
		key2x=key2
		for character in ch:
			if character in ch2:
				key2x=key2x.replace(character,' ')
			else:
				key2x=key2x.replace(character,'')
		words2=key2x.split()
		words=[]
		for word2 in words2:
			word=re.sub(r'(\W)(?=\1)', '', word2)
			word=word.lower()
			words.append(word)
		key = ' '.join(words)
		if len(key)>0:
			if key in mydict:
				mydict[key]+=mydict2[key2]
			else:
				mydict[key]=mydict2[key2]
		
	zeus=open(sys.argv[1],'w')
	for key,value in mydict.items():
		print(key,file=zeus)

def input_from_file():
	mydict={}
	with open(sys.argv[1],'r') as f:
		for row in csv.reader(f,delimiter='\t'):
			if len(row)>0:
				if row[0] in mydict:
					mydict[row[0]]+=1
				else:
					mydict[row[0]]=1

	with open('new_' + sys.argv[1],'w') as f:
		for key in sorted(mydict,key=mydict.get,reverse=True):
			words=key.split()
			words2=[]
			for word in words:
				word2=word.lower()
				words2.append(word2)
			words2=' '.join(str(word2) for word2 in words2)
			if words2 in mydict2:
				mydict2[words2]+=mydict[key]
			else:
				mydict2[words2]=mydict[key]

		for key in sorted(mydict2,key=mydict2.get,reverse=True):
			print(key,file=f)

if __name__=="__main__":
	input_from_file()
	replace_to_single()
