import scrapy
import re
import json
from html import unescape
import unicodedata
import pymongo
from bson.objectid import ObjectId
from datetime import datetime, timedelta

class FptSpider(scrapy.Spider):
    name = "fpt"
    allowed_domains = ["fptshop.com.vn"]
    start_urls = ["https://fptshop.com.vn/dien-thoai?sort=ban-chay-nhat&trang=1"]

    def __init__(self):
        self.client = pymongo.MongoClient('mongodb+srv://thiendihill181:A0YZHAJ9L4kxZfhb@cluster0.ys2zvmm.mongodb.net/')
        self.db = self.client['test']
        self.collection = self.db['phones']
        query = {"$and": [
                {"_id": {"$exists": True}},
                {"name": {"$exists": True}}
            ]}
        self.record = self.collection.find_one(query)
        
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, meta={
                "playwright": True, "playwright_include_page": True
            })
    def parse(self, response):
        print(response)
        next_page = "https://fptshop.com.vn/dien-thoai?sort=ban-chay-nhat&trang=4"
        yield scrapy.Request(next_page, callback=self.parse_total, meta={
                "playwright": True, "playwright_include_page": True
            }) 
        
    def parse(self, response):
        link_phone_fpts = []
        clean_names = []
        link_dict_fpt = {}
        for product in response.xpath("//div[@class='cdt-product-wrapper m-b-20']/div"):
            link_phone_fpt = product.xpath(".//div[@class='cdt-product__info']/h3/a/@href").get()
            name_phone_fpt = product.xpath(".//div[@class='cdt-product__info']/h3/a/text()").get()
            id_phone_fpt = product.xpath(".//div[@class='cdt-product__config']")
            id_phone_fpt = id_phone_fpt.xpath(".//div[@class='cdt-product__img-promo']/div/@id").get()
            if link_phone_fpt is not None and name_phone_fpt is not None:
                clean_name = name_phone_fpt.strip()
                id_phone = re.search(r'\d+', str(id_phone_fpt)).group()
                link_dict_fpt[clean_name] = {
                    "link": link_phone_fpt,
                    "id_phone": id_phone
                }
                # yield{
                #     "id": id_phone,
                #     "link_phone_fpt": link_phone_fpt,
                #     "name_phone_fpt": clean_name
                # }
            else:
                continue
            link_phone_fpts.append(link_phone_fpt)
            clean_names.append(clean_name)
        item = self.record["name"]
        if item in link_dict_fpt and link_dict_fpt[item]["id_phone"]:
            id_phone = link_dict_fpt[item]["id_phone"]
            yield scrapy.Request(
                url=response.urljoin(link_dict_fpt[item]["link"]),
                callback=self.parse_phone_page,
                meta={'id_phone': id_phone, "playwright": True, "playwright_include_page": True}
            )
        #    40069
                
    def parse_phone_page(self, response):
        id_phone = response.meta.get('id_phone')
        print(id_phone)
        next_page = "https://fptshop.com.vn/api-data/comment/api/Reviews?productId=" + str(id_phone) + "&pageIndex=" + str(1) + "&sort=0"
        print(next_page)
        for product in response.xpath("//div[@class='card re-card st-card']/div/table/tbody/tr"):
            specificatons = product.xpath("td[1]/text()").get()
            parameter = product.xpath("td[2]/text()").get()
            yield{
                        "parameter": parameter
                    }
            data_para = {specificatons: parameter}
            self.collection.update_one( 
                        {"_id": {"$exists": True}}, 
                        {"$push": {
                            "specificatons": {'$each': [data_para]}}})
    #     total_rating = response.xpath(".//div[@class='star']/div[2]/text()").get()
    #     total_cmt = response.xpath(".//div[@class='star']/div[3]/text()").get()
    #     five_rating = response.xpath(".//div[@class='progress-block']/div[1]/span[3]/text()").get()
    #     item = {
    #         "total_rating": total_rating,
    #         "total_cmt": total_cmt,
    #         "five_rating": five_rating
    #     }
    #     self.collection.update_one( 
    #                         {"_id": {"$exists": True}}, 
    #                         {"$set": {
    #                             "overview_fpt":item}})
    #     yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse_cmt_phone,  meta={'id_phone': id_phone, 
    #         "playwright": True, "playwright_include_page": True
    #         },)
    
    
    # def parse_cmt_phone(self, response):
    #     page = response.meta["playwright_page"]
    #     print(page)
    #     id_phone = response.meta.get('id_phone')
    #     string_object = response.body.decode('utf-8')
    #     string_object = re.sub('<[^<]+?>', '', string_object)
    #     data = json.loads(string_object)
    #     # def convert_date(date_str):
    #     #     if "giờ" in date_str:
    #     #         hours = int(date_str.split()[0])
    #     #         date = datetime.now() - timedelta(hours=hours)
    #     #     elif "ngày" in date_str:
    #     #         days = int(date_str.split()[0]) 
    #     #         date = datetime.now() - timedelta(days=days)  
    #     #         return None
        
    #     #     if date:
    #     #         formatted_date = date.strftime("%d/%m/%Y")
    #     #     else:
    #     #         formatted_date = ""
    #         # return formatted_date
    #     reviews = data['datas']['listReview']
        
    #     if  len(reviews) > 0:
    #         for review in reviews:
    #             comment = review['commentCustomer']
    #             date = review['convertDate']
    #             yield{
    #                     "comment":comment,
    #                     "date": date
    #                 }
    #             data = {
    #                     "comment":comment,
    #                     "date": date
    #                 }
    #             self.collection.update_one( 
    #                         {"_id": {"$exists": True}}, 
    #                         {"$push": {
    #                             "data": {'$each': [data]}}})
    #         id_phone = response.meta.get('id_phone')
    #         print(id_phone)
    #         data = response.meta.get('data', [])
    #         i = response.meta.get('page', 1) + 1
    #         print(i)
    #         if i <= 100:
    #             next_page = "https://fptshop.com.vn/api-data/comment/api/Reviews?productId=" + str(id_phone) + "&pageIndex=" + str(i) + "&sort=0"
    #             yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse_cmt_phone,  meta={'id_phone': id_phone, 'data': data, 'page': i,
    #                 "playwright": True, "playwright_include_page": True
    #                 },)
        

