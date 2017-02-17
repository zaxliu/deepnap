import os
print "Unzipping data files:"
os.system('cd data/; tar xvf traces_kdd.tar.gz; cd ..')
print "Unzipping compiled indexed log:"
os.system('cd data/; tar xvPf log_index.tar.gz; cd ..')
print "Downloading uncompiled indexed log:"
files = ['figure10.tar.gz',
         'figure11.tar.gz',
         'figure2.tar.gz',
         'figure3.tar.gz',
         'figure4.tar.gz',
         'figure5.tar.gz',
         'figure9.tar.gz']
for file in files:
    os.system('cd experiments/log/; wget network.ee.tsinghua.edu.cn/datasets/deepnap/{f}; tar xvf {f}'.format(f=file))
