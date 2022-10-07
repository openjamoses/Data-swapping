Authors_obj = {'foutse khomh':'YYXb3KIAAAAJ', 'Heng Li':'4tzkGvMAAAAJ', 'G Antoniol':'136elhQAAAAJ',
               'J Cheng':'ebhZJVEAAAAJ', 'B Adams':'XS9QH_UAAAAJ', 'M Fokaefs':'CFKjCQQAAAAJ', 'M Hamdaqa':'axhklvMAAAAJ',
               'M Lamothe':'N-SSIvIAAAAJ', 'W Shang':'ZnERyl4AAAAJ', 'Z Sharafi':'_WX4ZGYAAAAJ', 'H Sahraoui':'xsUkTCEAAAAJ',
               'E Syriani':'iQlrBWkAAAAJ', 'Michalis Famelis':'oqHf2lQAAAAJ', 'Emad Shihab':'kF76KPAAAAAJ', 'Tse-Hsun (Peter) Chen':'gUBD7x0AAAAJ',
               'Jinqiu Yang':'DpY9rhEAAAAJ', 'Weiyi (Ian) Shang': 'ZnERyl4AAAAJ', 'ZM Jiang':'dbzTZhcAAAAJ',
               'Ulrich Aivodji':'47kuuqIAAAAJ', 'Ali Ouni':'RHH5rtcAAAAJ', 'Mohammed Sayagh':'c2P4vhIAAAAJ', 'Ahmed E. Hassan':'9hwXx34AAAAJ',
               'Ying Zou':'pCYlknMAAAAJ', 'Lionel C. Briand':'Zj897NoAAAAJ', 'Maleknaz Nayebi':'uSWqDhwAAAAJ',
               'Lei Ma':'xsfGc58AAAAJ', 'Shane McIntosh':'FxUqGoUAAAAJ'}

list_id = []

import csv
import urllib.request, json

def read_url(url):
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data
class Publications:
    @staticmethod
    def search_by_name(author_name):
        #foutse%20khomh
        url = 'https://serpapi.com/search.json?engine=google_scholar&q=software+author:'+author_name+'&api_key=844d57eeef78649663e94817692c0056591fb74b446af97dbb2a04606e40a56c'
        json_data = read_url(url)
        total_results = json_data['total_results']
        for obj_results in json_data['organic_results']:
            title = obj_results['title']
            link = obj_results['link']
            snippet = obj_results['snippet']
            cited_by = obj_results['cited_by']['total']
            title = obj_results['title']
            title = obj_results['title']
            title = obj_results['title']
        return read_url(url)
    # todo: Author API : https://serpapi.com/search?engine=google_scholar_author&author_id=j6ucyOAAAAAJ
    @staticmethod
    def search_author_id():
        data_file = open(data_path + 'Publications_15-07-2022.csv', mode='w', newline='',
                                encoding='utf-8')
        data_writer = csv.writer(data_file)
        data_writer.writerow(
            ['authors', 'publication', 'year', 'citations', 'title', 'link', 'searched_semla_professor', 'affiliations', 'website', 'author_id', 'research_interest'])
        list_title = []
        for key, author_id in Authors_obj.items():
            url = 'https://serpapi.com/search?engine=google_scholar_author&author_id='+author_id+'&api_key=844d57eeef78649663e94817692c0056591fb74b446af97dbb2a04606e40a56c'
            json_data = read_url(url)
            name = json_data['author']['name']
            affiliations = json_data['author']['affiliations']
            website = ''
            try:
                website = json_data['author']['website']
            except:
                pass


            list_interest = []
            for interest_obj in json_data['author']['interests']:
                list_interest.append(interest_obj['title'])
            articles = json_data['articles']
            print(author_id, name, affiliations, len(articles))

            for article_obj in articles:
                title = article_obj['title']
                authors = article_obj['authors']
                publication = ''
                try:
                    publication = article_obj['publication']
                except:
                    pass
                year = article_obj['year']
                cited_number = article_obj['cited_by']['value']
                link = article_obj['cited_by']['link']
                if not title in list_title:
                    data_writer.writerow(
                        [authors, publication, year, cited_number, title, link, name, affiliations, website, author_id, list_interest])

        data_file.close()

data_path = '/Volumes/Cisco/Summer2022/SEMLA/Publications/'
if __name__ == '__main__':

    #json_data = Publications.search_by_name('foutse%20khomh')
    #print(json_data)
    Publications.search_author_id()
