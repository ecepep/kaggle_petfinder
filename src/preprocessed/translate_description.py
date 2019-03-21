'''
Created on Jan 21, 2019

@author: ppc

To many descriptions to work (Google block googletrans request)

translate all description to english 
through the UNOFFICIAL reversed engineered googletrans api
'''


from classification.util import getTrainTest2
import re
from googletrans import Translator
import pandas as pd
from eventlet.timeout import Timeout

pathToAll = "../all" # path to dataset dir
trans_dir = "/preprocessed/description_translation/"
# read train, test csvs, set unknown to NA and shuffle
train, test = getTrainTest2(pathToAll)
# train =  train.iloc[1:3, :]
# test =  test.iloc[1:3, :]
# print(test)
# translating description
def des_translation(des, maxTrans = 100):
    '''
    translate description to english
    :param des:
    :param maxTrans above a certain threshold google send to googletrans;
    '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n<html>\n<head><meta http-equiv="content-type" content="text/html; charset=utf-8"><meta name="viewport" content="initial-scale=1"><title>https://translate.google.com/translate_a/single?client=t&amp;sl=auto&amp;tl=en&amp;hl=en&amp;dt=at&amp;dt=bd&amp;dt=ex&amp;dt=ld&amp;dt=md&amp;dt=qca&amp;dt=rw&amp;dt=rm&amp;dt=ss&amp;dt=t&amp;ie=UTF-8&amp;oe=UTF-8&amp;otf=1&amp;ssel=0&amp;tsel=0&amp;tk=280439.182453&amp;q=Finding+new+owner+.</title></head>\n<body style="font-family: arial, sans-serif; background-color: #fff; color: #000; padding:20px; font-size:18px;" onload="e=document.getElementById(\'captcha\');if(e){e.focus();}">\n<div style="max-width:400px;">\n<hr noshade size="1" style="color:#ccc; background-color:#ccc;"><br>\n<div style="font-size:13px;">\nOur systems have detected unusual traffic from your computer network.  Please try your request again later.  <a href="#" onclick="document.getElementById(\'infoDiv0\').style.display=\'block\';">Why did this happen?</a><br><br>\n<div id="infoDiv0" style="display:none; background-color:#eee; padding:10px; margin:0 0 15px 0; line-height:1.4em;">\nThis page appears when Google automatically detects requests coming from your computer network which appear to be in violation of the <a href="//www.google.com/policies/terms/">Terms of Service</a>. The block will expire shortly after those requests stop.<br><br>This traffic may have been sent by malicious software, a browser plug-in, or a script that sends automated requests.  If you share your network connection, ask your administrator for help &mdash; a different computer using the same IP address may be responsible.  <a href="//support.google.com/websearch/answer/86640">Learn more</a><br><br>Sometimes you may see this page if you are using advanced terms that robots are known to use, or sending requests very quickly.\n</div><br>\n\nIP address: 88.173.120.20<br>Time: 2019-01-21T10:08:03Z<br>URL: https://translate.google.com/translate_a/single?client=t&amp;sl=auto&amp;tl=en&amp;hl=en&amp;dt=at&amp;dt=bd&amp;dt=ex&amp;dt=ld&amp;dt=md&amp;dt=qca&amp;dt=rw&amp;dt=rm&amp;dt=ss&amp;dt=t&amp;ie=UTF-8&amp;oe=UTF-8&amp;otf=1&amp;ssel=0&amp;tsel=0&amp;tk=280439.182453&amp;q=Finding+new+owner+.<br>\n</div>\n</div>\n</body>\n</html>\n'
    maxTrans define the number of trans on the same Translator object 
    '''
    translator = Translator()
    translations = list()
    
    count = 0
    for i in des:
        print(count)
        count +=1
        if (type(i) is str) and (len(i) > 0):
            first_attempt = True
            done =  False
            while not done:
                print(i)
                try:
                    ts = translator.translate(i, dest = 'en').text
                    translations.append(ts)
                    done = True
                except:
                    # for good translation of some specific char, would be needing something to sanitize str for json
                    formated = re.sub(r'[^A-Za-z1-9\.\s]', '', i)
                    try:
                        ts = translator.translate(formated, dest = 'en').text
                        translations.append(ts)
                        done = True
                    except:
                        if first_attempt:
                            first_attempt = False
                            translator = Translator()
                        else: raise 'new translator method failed'
        else:
            translations.append("")
    return translations

timeout = Timeout(900)
try:
    train_ts = des_translation(train['Description'])
    test_ts = des_translation(test['Description'])
finally:
    timeout.cancel()
 
sep=","
assert train_ts.shape[0]
train_ts = [s.replace(sep, '') for s in train_ts]
test_ts = [s.replace(sep, '') for s in test_ts]
 
train_translations = pd.DataFrame(data={'PetID': train["PetID"], 
                                        'Description': train["Description"],
                                        'Translation': train_ts })
  
test_translations = pd.DataFrame(data={'PetID': test["PetID"], 
                                        'Description': test["Description"],
                                        'Translation': test_ts })
 
train_translations.to_csv(pathToAll + trans_dir + "train.csv", sep=sep)
test_translations.to_csv(pathToAll + trans_dir + "test.csv", sep=sep)

print("Done")