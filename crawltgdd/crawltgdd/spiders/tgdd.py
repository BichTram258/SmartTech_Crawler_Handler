import scrapy
from scrapy.crawler import CrawlerProcess
import pymongo
from bson.objectid import ObjectId
import re

class TgddSpider(scrapy.Spider):
    name = "tgdd"
    allowed_domains = ["www.thegioididong.com"]
    start_urls = ["https://www.thegioididong.com/dtdd"]

    def __init__(self):
        self.client = pymongo.MongoClient('mongodb+srv://thiendihill181:A0YZHAJ9L4kxZfhb@cluster0.ys2zvmm.mongodb.net/')
        self.db = self.client['test']
        self.collection = self.db['phones']
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, meta={
                "playwright": True, "playwright_include_page": True
            },)
    def parse(self, response):
        next_page = "https://www.thegioididong.com/dtdd#c=42&o=17&pi=5"
        yield scrapy.Request(next_page, callback=self.parse_total, meta={
                "playwright": True, "playwright_include_page": True
            },)
    def parse_total(self, response):
        link_phone_tgdds = []
        clean_names = []
        print(response)
        visited_links = set()
        visited_names = set()
        for product in response.xpath(".//ul[@class='listproduct']/li"):
            link_phone_tgdd = product.xpath(".//a[@class='main-contain ']/@href").get()
            name_phone_tgdd = product.xpath(".//a[@class='main-contain ']/h3/text()").get()
            clean_name = name_phone_tgdd.strip()
            if link_phone_tgdd in visited_links or clean_name in visited_names:
                continue
            visited_links.add(link_phone_tgdd)
            visited_names.add(clean_name)
            link_phone_tgdds.append(link_phone_tgdd)
            clean_names.append(clean_name)
            yield{
                "link_phone_tgdd": link_phone_tgdd,
                "name_phone_tgdd": clean_name
            }
        
        link_dict_tgdd = {name: link for name, link in zip(clean_names, link_phone_tgdds)}
        item = "Samsung Galaxy Z Flip4 5G"
        if link_dict_tgdd[item]:
            next = "https://www.thegioididong.com" + link_dict_tgdd[item]
            yield scrapy.Request(url=response.urljoin(next), callback=self.parse_phone_page, meta = {"link": next, "playwright": True,
        "playwright_context": "awesome_context"})

    def parse_phone_page(self, response):
        link = response.meta.get('link')
        total_rating = response.xpath(".//div[@class='box-star']/div[1]/p/text()").get()
        total_cmt = response.xpath(".//div[@class='box-star']/div[1]/a/@data-total").get()
        five_rating = response.xpath(".//ul[@class='rate-list']/li[1]/span/text()").get()
        item = {
            "total_rating": total_rating,
            "total_cmt": total_cmt,
            "five_rating": five_rating
        }
        self.collection.update_one( 
                            {"_id": ObjectId("64a3dd71da3639eb81f42366")}, 
                            {"$set": {
                                "overview_tgdd":item}})
    #     cmtphone_tgdd = response.xpath(".//div[@class='rt-list']/div/a/@href").get()
    #     next_page = str(link) + "/" + str(cmtphone_tgdd)
    #     print(next_page)
    #     yield scrapy.Request(next_page, callback=self.parse_cmtphone_page, meta={
    #            "playwright": True, "playwright_include_page": True
    #         },)
        
    # def parse_cmtphone_page(self, response):
    #     for product in response.xpath(".//div[@class='rt-list']/ul/li"):
    #         cmt_phone_tgdd = product.xpath(".//div[@class='cmt-content ']/p/text()").get()
    #         date_buy = product.xpath(".//div[@class='cmt-command']/span/text()").get()
    #         if cmt_phone_tgdd is not None:
    #             # date_buy = date_buy[-1]
    #             yield {
    #                 "comment": cmt_phone_tgdd,
    #                 "date": date_buy
    #             }
                
    #             data = {
    #                         "comment":cmt_phone_tgdd,
    #                         "date": date_buy
    #                     }
                   
    #             self.collection.update_one( 
    #                         {"_id": ObjectId("64a3dd71da3639eb81f42366")}, 
    #                         {"$push": {
    #                             "data": {'$each': [data]}}})
    #         else:
    #             continue
    #     data = response.meta.get('data', [])
    #     i = response.meta.get('page', 1) + 1
    #     print(i)
    #     if i <= 20:
    #         link_next = response.xpath(".//ul[@class='breadcrumb-rating']/li/a/@href").get()
    #         print(link_next)
    #         next_page = "https://www.thegioididong.com" + str(link_next) + "/danh-gia?page=" + str(i)
    #         yield scrapy.Request(next_page, callback=self.parse_cmtphone_page, meta={'data': data, 'page': i,
    #             "playwright": True, "playwright_include_page": True
    #         },)
        
    
        
        
