import re
import urllib2
import urllib
from bs4 import BeautifulSoup
import pandas
import csv
import os.path



def search_allaboutbirds(bird_name):

    trial_name = bird_name.replace("'", "")
    
    url = 'https://www.allaboutbirds.org/guide/' + trial_name + '/id'
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page.read(page), 'html.parser')
    imgdiv = soup.find('div', { 'id' : 'id_glamor' })
    
    if imgdiv == None:
        print 'Did not find image of ' + bird_name
        return None

    image = imgdiv.findAll("img")[0].get('src')
    
    ipath = 'https://www.allaboutbirds.org/' + image

    return ipath

def search_wikipedia(bird_name):

    try: 
        url = 'https://en.wikipedia.org/wiki/' + bird_name
        page = urllib2.urlopen(url)


        soup = BeautifulSoup(page.read(page), 'html.parser')
        imgtab = soup.find('table', { 'class' : 'infobox biota' })

        imgrow = imgtab.findAll('tr')[1].findAll('img')[0]
        ipath = 'http:' + imgrow.get('src')

    except:
        print 'Cannot find image ' + bird_name
        ipath = None
    
    return ipath 




taxonomy = pandas.read_csv('../data/taxonomy.csv', header=0)
bird_dict = dict(zip(taxonomy['SCI_NAME'], taxonomy['PRIMARY_COM_NAME']))

taxonomy = pandas.read_csv('../data/bird_names.csv', header=0)

with open('../data/bird_names.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    sci_names = reader.next()

for sci_name in sci_names:
    bird_name = bird_dict[sci_name]

    filename = '../data/bird_images/' + bird_name + '.jpg'
    
    if not os.path.isfile(filename):

        print 'no image for ' + bird_name + ' yet...'
        #ipath = search_wikipedia(bird_name)
       
        #if ipath != None:
        #    urllib.urlretrieve(ipath, '../data/bird_images/wikipedia/' + bird_name + '.jpg')



