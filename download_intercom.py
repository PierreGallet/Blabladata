# coding:utf8
import os, json, pickle, time, csv, re

def remove_tags(text):

    TAG_RE = re.compile(r'<[^>]+>')

    if text is None:
        return ''
    return TAG_RE.sub(' ', text)

def convert_to_csv():
    with open('./output.pkl', 'rb') as f:
        r = pickle.load(f)

        with open('./output.csv', 'r+') as new_csv:
            writer = csv.DictWriter(new_csv, fieldnames=['label', 'sentence'], delimiter=';')
            writer.writeheader()
            print(type(r))
            for i in range(len(r)):
                conv = json.loads(r[i])
                subject = remove_tags(conv["conversation_message"]["subject"]).encode('utf8')
                messages = []
                messages.append(remove_tags(conv["conversation_message"]["body"]).encode('utf8'))
                for j in range(len(conv["conversation_parts"]["conversation_parts"])):
                    messages.append(remove_tags(conv["conversation_parts"]["conversation_parts"][j]["body"]).encode('utf8'))
                for k in range(len(messages)):
                    writer.writerow({'label': subject, 'sentence': messages[k]})
                writer.writerow({'label': "___", 'sentence': "___"})
                print(i)


def get_user_params():

    user_params = {}

    # get user input params
    user_params['appId']  = raw_input( '\nappId: ' )
    user_params['apiKey']  = raw_input( '\napiKey: ' )

    # apply defaults
    if user_params['appId']  == '':
        user_params['appId'] = 'appId'
    if user_params['apiKey']  == '':
        user_params['apiKey'] = 'apiKey'

    return user_params


def dump_user_params( user_params ):

    # dump user params for confirmation
    print('appId:    '   + user_params['appId'])
    print('apiKey:    '   + user_params['apiKey'])
    return

# commande shell
# curl https://api.intercom.io/conversations?page=1 -X GET -u ruhyckjd:bec503d8eb8cf8dd79b44c2aa762fa0c642a2dc8 -H 'Accept: application/json'
def get_data(appId, apiKey):

    id = []
    for page_id in range(500):
        print('page_id:', page_id)
        cmd = ("curl https://api.intercom.io/conversations?page="+str(page_id)+" -X GET -u "+appId+":"+apiKey+" -H 'Accept: application/json'")
        response = os.popen(cmd).read()
        voila = json.loads(response)
        for i in range(len(voila["conversations"])):
            try:
                id.append(voila["conversations"][i]["id"])
            except:
                pass

    print(id)
    print('nombre de conversations:', len(id))
    result = []
    i = 1
    for j in id:
        try:
            cmd = ("curl https://api.intercom.io/conversations/"+str(j)+" -X GET -u "+appId+":"+apiKey+" -H 'Accept: application/json'")
            response = os.popen(cmd).read()
            result.append(response)
            print('one more')
        except:
            print('error')
            pass
        i += 1

    with open('./output.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    # user_params = get_user_params()
    # dump_user_params(user_params)

    # From Us
    # appId = "ruhyckjd"
    # apiKey = "bec503d8eb8cf8dd79b44c2aa762fa0c642a2dc8"

    # from Nestor
    appId = "t71de6dd"
    apiKey = "ro-7f8df8b9e666bbb1a7382db0846e9ff887596579"


    get_data(appId, apiKey)
