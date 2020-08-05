import scrapy
#!pip install nltk

#crapy runspider G1Spider.py --nolog - o atu.json
class G1spiderSpider(scrapy.Spider):
    name = 'G1Spider'
    allowed_domains = ['g1.globo.com']
    start_urls = ['http://g1.globo.com/']

    def parse(self, response):
        page_title = response.css('title::text').extract_first()
        news_list = response.css('.bastian-page .bastian-feed-item')    
        print(page_title)
        print()
        #print(news_list)
        for news in news_list:
            title = news.css ('.feed-post-link::text').extract_first()
            description = news.css('.feed-post-body-resumo::text').extract_first()
            image_url = news.css('.bstn-fd-picture-image::attr(src)').extract_first()
            link = news.css('.feed-post-link::attr(href)').extract_first()
            cat = news.css('.feed-post-metadata .feed-post-metadata-section::text').extract_first()

        #print({'title':title, 'description':description, 'image_url':image_url, 'link':link})
        #print({'description':description})
            yield({'title':title, 'description':description, 'image_url':image_url, 'link':link,'categoria':cat})
           
        next_page = response.css('.load-more a::attr(href)').extract_first()
       
        if (page_title[-2:] != "99" ) :
            yield response.follow(next_page, self.parse)
                
