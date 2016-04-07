from glob import glob
from os import system 
for filename in glob('*.ipynb'):
    command="jupyter nbconvert --to slides '%s'"%filename
    print command
    system(command)
    name=filename[:-6] # chop off the ipynb
    print "1. [%s](/DSE230/Notebook-decks/%s.slides.html)"%(name,name)


