import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
x = []
y = []
with open(sys.argv[1], "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # i=0
    for row in reader:
      if(int(row[1])!=0):
      	if(int(row[1])>5):
					x.append(row[0])
					y.append(int(row[1]))
        else:
					y.append(int(row[1]))
					x.append("")
      else:
				x.append("")
				y.append(0)
y = np.asarray(y)
ind = np.arange(len(x))  # the x locations for the groups
width = 0.01       # the width of the bars

fig, ax = plt.subplots()
width = 1/1.5
plt.bar(ind,y, width, color="blue")
fig.subplots_adjust(bottom=0.2)

#ax.ylim(0,200)
ax.set_ylabel('Y axis')
ax.set_title('X axis')
ax.set_xticks(ind + width)
ax.set_xticklabels(x, rotation='vertical')

plt.savefig(sys.argv[1] + '.pdf')
# plt.show()
