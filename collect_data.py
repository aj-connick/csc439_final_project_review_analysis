
import json
import os
import subprocess
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import math

#movie_web_page = "https://www.rottentomatoes.com/browse/dvd-streaming-all/"
'''
olded function for breaking down javascript function
'''
def get_movie_reviews(web_page):
    #domtree = requests.get(movie_web_page)
    domtree ="null"
    tree = domtree.text
    movies = []
    web = BeautifulSoup(domtree.content, "html.parser")
    movies = web.find_all('script')
    movie_names = movies[39].text
    print(type(movie_names))
    clean_names = movie_names[movie_names.index("[{"):movie_names.index("}]")]
    a = clean_names.split(",")
    links = []
    for x in a:
        if "url" in x:
            links.append(x.strip("\"url\":"))

    print(len(links))

movie_web_imdbb = "https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm"
nmber_of_ratings ="https://www.imdb.com/chart/moviemeter/?sort=nv,desc&mode=simple&page=1"
'''
runs over domtree from IMDB and bilds a list of movies sing link as the entry point
'''
def get_movie_names(movie_web_imdbb):
    domtree = requests.get(movie_web_imdbb)
    web = BeautifulSoup(domtree.content, "html.parser")
    movies = web.find_all("td",class_="titleColumn")
    build_movie_links = []
    for i in movies:
        a = i.find_all('a')
        path = a[0].get('href')
        title = path.split("/")[2]
        build_movie_links.append(title)
    return build_movie_links
'''
bilds a link to a given movie title
'''
def build_link(title):
    #https://www.imdb.com/title/tt11138512/reviews?ref_=tt_urv
    return "https://www.imdb.com/title/"+title+"/reviews?ref_=tt_urv"
'''
runs over the domtree of each movie link and
build file containng movie reviews
'''
def get_reviews(titles):
    fp = open("movie_reviews_run_7.txt","w")
    for i in titles:#movies
        link = build_link(i)
        domtree = requests.get(link)
        web = BeautifulSoup(domtree.content, "html.parser")
        review = web.find_all("div",class_="review-container")
        for k in review:#each review
            o = k.find_all('span')
            rating = o[1].text
            rev = k.find_all('div',class_="text show-more__control")
            review2 = (rev[0].text).strip()
            save_rev = rating+ " " + review2.rstrip()
            fp.write(save_rev)
            fp.write("\n")
'''
converst numaric vales to labels
'''
def clean_data_save(file):
    fp = open(file,errors='ignore')
    rescale = []
    label = ""
    text = ""
    fp2 = open("data_set_full_2.txt","a")
    for i in fp:
        o = i.split(" ")[0]
        label = "no"
        text = "no"
        if '0' == o or '1' == o or '2' == o or '3' == o:
            label = 'bad'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        elif '4' == o or '5' == o or '6' == o or '7' == o:
            label = 'ok'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        elif '8' == o or '9' == o or '10' == o:
            label = 'good'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        if label != "no" and text != "no":
            save_rev = label+ " " + text.rstrip()
            fp2.write(save_rev)
            fp2.write("\n")
    return rescale
'''
cuts data into dev test and train
writes each data set back to a file
'''
def cut_data_set(file):
    fp = open(file)
    #size  = len(fp.readline())
    size = 0
    for k in fp:
        size +=1
    fp.close()
    fp = open(file)
    dev = open("data/movie_dev.txt","w")
    test = open("data/movie_test.txt","w")
    train = open("data/movie_train.txt","w")
    #20/20/60 +or- 1
    dev_size = math.floor(size * .20)
    test_size = math.floor(size * .20)
    train_size = math.floor(size * .60)
    fp_pointer = 0
    print(train_size)
    print(test_size)
    print(dev_size)
    print(size)
    for i in fp:
        if fp_pointer <= train_size:
            train.write(i)
        elif fp_pointer <= train_size + test_size:
            test.write(i)
        else:
            dev.write(i) 
        fp_pointer += 1
'''
counts the nmber of each type of review
and prints to console
'''
def movie_stats(file):
    fp = open(file)
    data = []
    good = 0
    bad = 0
    ok = 0
    for i in fp:
        k = i.split()
        label = k[0]
        if label == "good":
            good +=1
        elif label == "ok":
            ok += 1
        elif label == "bad":
            bad += 1
    print("======="+file+"=======" )
    print("Good = " + str(good))
    print("Ok = " + str(ok))
    print("Bad = " + str(bad))

    return data


def main():
    file_train = "data/movie_train.txt"
    file_test = "data/movie_test.txt"
    file_dev = "data/movie_dev.txt"
    movie_stats(file_train)
    movie_stats(file_test)
    movie_stats(file_dev)
    #title = []

    #title = get_movie_names(nmber_of_ratings)
    #get_reviews(title)
if __name__=="__main__":
    main()


