# coding: utf8
from __future__ import print_function
import os, urllib, json, shutil, sys, time, csv
from HTMLParser import HTMLParser


"""
Translating english to french for .txt and .csv format
(.txt need to be in the folder datadir with on line per file)
(.csv need to be ; separated with two columns 'label' and 'sentence')
"""

def get_user_params():

    user_params = {}

    # get user input params
    user_params['Input']  = raw_input( '\nInput file [./corpus.csv]: ' )

    # apply defaults
    if user_params['Input']  == '':
        user_params['Input'] = './corpus.csv'


    return user_params


def dump_user_params( user_params ):

    # dump user params for confirmation
    print('Input:    '   + user_params['Input'])
    return

def parser_inputurl_outpututf(input):
    input = input.decode('utf-8')
    parser = HTMLParser()
    input = input.encode('ascii', errors='xmlcharrefreplace')
    input = parser.unescape(input)
    input = input.encode('utf-8')
    return input


def save_txt(txt, path):
    with open(path, "w+") as file:
        file.write(txt)
        file.close()
    return txt, path

def save_csv(label, sentence, path, fieldnames=['label', 'sentence']):
    with open(path, "w+") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerow({'label': label, 'sentence': sentence})

def translate_csv_with_google(datadir):
    datadir_fr = '.'.join(datadir.split('.')[:-1])
    datadir_fr = datadir_fr+'_fr.csv'
    APIKEY = "AIzaSyADnr0i9zvLa8jQ6xP0fP7oiPYHeukfzys"
    with open(datadir_fr, 'w+') as new_csv:
        writer = csv.DictWriter(new_csv, fieldnames=['label', 'sentence'], delimiter=';')
        writer.writeheader()
        with open(datadir, 'rb') as f:
            reader = csv.DictReader(f, fieldnames=['label', 'sentence'], delimiter=';')
            i = 0
            j = 0
            for row in reader:
                # to respect API limitation (100k characters per 100s so aprox. 20 files per 5s)
                i += 1
                if i % 10 == 0:
                    print(str(i)+' files treated')
                    time.sleep(10)
                txt = row['sentence']
                label = row['label']
                txt = urllib.quote(txt, safe='')
                cmd = ('curl -s https://www.googleapis.com/language/translate/v2?key='+APIKEY+'\&source=en\&target=fr\&q='+txt)
                # getting response from GoogleTrad API in Json format (but as a str here)
                trad = os.popen(cmd).read()
                try:
                    # loading the str in a json format
                    trad = json.loads(trad)
                    # selecting the relevant data
                    trad = trad["data"]["translations"][0]["translatedText"]
                    # encoding the data in utf-8 so that it can be printed
                    trad = str(trad.encode('utf-8'))
                    # retreaving the HTML entities sur as &#39; and &quot;
                    trad = parser_inputurl_outpututf(trad)
                    # writing the trad in a new file
                    writer.writerow({'label': label, 'sentence': trad})
                except:
                    print('RowError:'+str(i)+'\nGoogleTrad API answer:'+str(trad))
                    j += 1

    return print(str(i-j)+' files translated in french')


def translate_txt_with_google(datadir):
    datadir_fr = datadir+'_fr'
    try:
        shutil.rmtree(datadir_fr)
    except:
        pass
    os.mkdir(datadir_fr)

    APIKEY = "AIzaSyADnr0i9zvLa8jQ6xP0fP7oiPYHeukfzys"
    files = os.listdir(datadir)
    j = 0

    for i in range(len(files)):
        # to respect API limitation (100k characters per 100s so aprox. 20 files per 5s)
        if i%10 == 0:
            print(str(i)+' files treated')
            time.sleep(9)
        with open(datadir+'/'+files[i]) as f:
            txt = urllib.quote((f.readlines()[0]), safe='')
            cmd = ('curl -s https://www.googleapis.com/language/translate/v2?key='+APIKEY+'\&source=en\&target=fr\&q='+txt)
            # getting response from GoogleTrad API in Json format (but as a str here)
            trad = os.popen(cmd).read()
            try:
                # loading the str in a json format
                trad = json.loads(trad)
                # selecting the relevant data
                trad = trad["data"]["translations"][0]["translatedText"]
                # encoding the data in utf-8 so that it can be printed
                trad = str(trad.encode('utf-8'))
                # retreaving the HTML entities sur as &#39; and &quot;
                trad = parser_inputurl_outpututf(trad)
                # writing the trad in a new file
                save_txt(trad, datadir_fr+'/'+str(i)+'.txt')
            except:
                print('FileError:'+str(files[i])+'\nGoogleTrad API answer:'+str(trad))
                j += 1

    return print(str(i-j)+' files translated in french')



if __name__ == '__main__':
    user_params = get_user_params()
    dump_user_params(user_params)
    datadir = user_params['Input']
    print('Datadir:', datadir)
    if datadir.split('.')[-1] == 'csv':
        translate_csv_with_google(datadir)
    else:
        translate_txt_with_google(datadir)
