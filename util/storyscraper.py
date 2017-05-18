#!../venv/bin/python

import argparse
import json
import logging as log
import os
import random
import requests
import string
import sys
import time

from bs4 import BeautifulSoup

#url without trailing slash
BASE_URL = "http://m.fanfiction.net"

def getGenreOverviewURLs():
	categories = ["anime", "book", "cartoon", "comic", "game", "misc", "movie", "play", "tv"]
	pages = [1] + [c for c in string.ascii_lowercase]

	urls = []
	for category in categories:
		for page in pages:
			urls.append(BASE_URL + "/{}/?l={}".format(category, page))

	return urls


def getUniverseURLs(genreURL):
	html = requests.get(genreURL).text
	dom = BeautifulSoup(html, "lxml")

	urls = []
	for link  in dom.select(".bs a"):
		try:
			urls.append(BASE_URL + link["href"])
		except KeyError:
			log.error("Failed on: {}".format(link))

	return urls

def getChapterURLs(universeURL, skip = 0):
	pageNumber = skip + 1
	while True:
		log.debug("Trying page {} on universe {}".format(pageNumber, universeURL))

		request = requests.get(universeURL + "/?_g1=5&lan=1&r=10&p={}".format(pageNumber))
		html = request.text

		if "No entries found with current filters." in html:
			break

		dom = BeautifulSoup(html, "lxml")
		possibleLinks = dom.select(".bs a")

		if len(possibleLinks) == 0:
			break

		urls = []
		for link in possibleLinks:
			target, storyid, rest = link["href"][1:].split("/", 2)

			if not target == "s":
				continue

			page, title = rest.split("/", 1)

			if page == "1":
				urls.append(BASE_URL + "/s/" + storyid + "/1")
				continue

			for otherPage in range(2, int(page) + 1):
				urls.append("{}/s/{}/{}".format(BASE_URL, storyid, otherPage))

		pageNumber += 1
		yield urls

def getStory(storyURL):
	html = requests.get(storyURL).text
	dom = BeautifulSoup(html, "lxml")

	story = dom.select("#storycontent")[0]
	storytext = [p.get_text() for p in story.find_all("p")]
	meta =  dom.select("#content")[0]

	return {"id": storyURL[len(BASE_URL):].replace("/", "-"),
	        "title": meta.find("b").text,
	        "author": meta.find("a").text,
	        "genre": meta.find_all("a")[1].text,
	        "universe": meta.find_all("a")[2].text,
	        "genreURL": meta.find_all("a")[1]["href"],
	        "universeURL": meta.find_all("a")[2]["href"],
	        "meta_raw": meta.get_text(),
	        "storytext": storytext}



def downloadAndSaveUniverseURLs(args):
	if not args.nocont:
		try:
			with open("univUrls.json", "r") as f:
				univUrls = json.load(f)
			with open("univUrlDict.json", "r") as f:
				univUrlsDict = json.load(f)
		except FileNotFoundError:
			univURls = []
			univUrlsDict = {}
	else:
		univURls = []
		univUrlsDict = {}

	overviewUrls = getGenreOverviewURLs()
	log.info("Generated {} overview URLS".format(len(overviewUrls)))
	sys.exit(1)

	for url in overviewUrls:
		if url in univUrlsDict or not url.startswith(args.urlFilter):
			continue

		newUrls = getUniverseUrls(url)
		univUrls += newUrls
		univUrlsDict[url] = newUrls

		with open("univUrls.json", "w") as f:
			json.dump(univUrls, f)

		with open("univUrlDict.json", "w") as f:
			json.dump(univUrlsDict, f)

		log.info("Finished " + url)

def downloadAndSaveStoryURLs(args):

	with open("univUrls.json", "r") as f:
		univUrls = json.load(f)
	with open("univUrlDict.json", "r") as f:
		univUrlsDict = json.load(f)

	if not args.nocont:
		try:
			with open("chapterUrls.json", "r") as f:
				chapterUrls = json.load(f)

			with open("chapterUrlDict.json", "r") as f:
				chapterUrlsDict = json.load(f)
		except FileNotFoundError:
			chapterUrls = []
			chapterUrlsDict = {}
	else:
		chapterUrls = []
		chapterUrlsDict = {}

	skip = args.skipPages
	seenStartUniverse = args.startAtUniverse == None
	saveIntervalIndex = 0
	try:
		for index, univUrl in enumerate(univUrls):

			if (univUrl in chapterUrlsDict and not univUrl == args.startAtUniverse) or not univUrl.startswith(args.urlFilter):
				continue

			if not seenStartUniverse and not univUrl == args.startAtUniverse:
				continue
			seenStartUniverse = True

			log.info("Finished {}/{} universes. Current: {}".format(len(chapterUrlsDict), len(univUrls), univUrl))

			for retry in range(0,2):
				try:
					for pages, newUrls in enumerate(getChapterURLs(univUrl, skip)):
						skip = 0
						chapterUrls += newUrls

						if pages == 0:
							chapterUrlsDict[univUrl] = []
						chapterUrlsDict[univUrl] += newUrls

						saveIntervalIndex += 1
						if saveIntervalIndex == 60:
							saveIntervalIndex = 0
							continuePages = pages + 1 + (0 if not univUrl == args.startAtUniverse else args.skipPages)
							log.info("Saving. ICE start at this universe {} and skip {} pages".format(univUrl, continuePages))
							with open("chapterUrls.json", "w") as f:
								json.dump(chapterUrls, f)

							with open("chapterUrlDict.json", "w") as f:
								json.dump(chapterUrlsDict, f)

				except requests.exceptions.ConnectionError:
					log.warning("Retrying due to a connection error")
					skip = pages + 1
					time.sleep(0.5)
				except Exception as e:
					log.error("Failed after {} pages due to: {}".format(pages, e))
					log.exception(e)
					with open("chapterUrls_failBackup.json", "w") as f:
						json.dump(chapterUrls, f)

					with open("chapterUrlDict_failBackup.json", "w") as f:
						json.dump(chapterUrlsDict, f)
					sys.exit(1) #we don't know what happened, better safe than sorry
				else:
					break #there was no error, we don't need to retry

		with open("chapterUrls.json", "w") as f:
			json.dump(chapterUrls, f)

		with open("chapterUrlDict.json", "w") as f:
			json.dump(chapterUrlsDict, f)
	except KeyboardInterrupt:
		log.info("Caught interrupt signal. Stopping, but writing json first, in case this was corrupted. Keep calm.")
		with open("chapterUrls_interruptBackup.json", "w") as f:
			json.dump(chapterUrls, f)

		with open("chapterUrlDict_interruptBackup.json", "w") as f:
			json.dump(chapterUrlsDict, f)


def downloadAndSaveStories(args):
	random.seed(42)
	with open("chapterUrls.json", "r") as f:
		chapterUrls = json.load(f)

	log.warning("This action is not safe on network file systems (NFS), due to os.open, sorry! Please check that you're not on an NFS.")
	allStories = set(chapterUrls)
	downloadedStories = set()
	while not len(allStories) == len(downloadedStories):
		chapterUrl = random.choice(chapterUrls)

		outFilePath = chapterUrl[len(BASE_URL):].replace("/", "-") + ".json"
		story = None
		outFile = None
		try:
			try:
				outFd = os.open(outFilePath, os.O_CREAT|os.O_WRONLY|os.O_EXCL)
			except OSError as e:
				log.debug("Chapter was downloaded by another thread")
				downloadedStories.add(chapterUrl)
				continue

			outFile = os.fdopen(outFd, "w")
			log.info("Downloading {} (>={}/{})".format(chapterUrl, len(downloadedStories), len(chapterUrls)))

			for retry in range(0,2):
				try:
					story = getStory(chapterUrl)
					json.dump(story, outFile)
					outFile.close()
					downloadedStories.add(chapterUrl)
					break
				except requests.exceptions.ConnectionError:
					log.warning("Retrying due to a connection error")
					time.sleep(0.5)

					if retry == 1:
						outFile.close()
						os.rm(outFilePath)
		except KeyboardInterrupt:
			if outFile and story:
				log.info("Caught interrupt signal. Stopping, but writing json first, in case this was corrupted. Keep calm.")
				outFile.seek(0)
				json.dump(story, outFile)
				outFile.close()
			elif outFd:
				if outFile:
					outFile.close()
				else:
					os.close(outFd)
				os.remove(outFilePath)
			break
		except Exception:
			if outFd:
				os.remove(outFilePath)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Scrape fanfiction.')
	parser.add_argument("--nocont", action = "store_true", help = "Don't continue scraping, even if files exist")
	parser.add_argument("--universeURLs", action = "store_true", help = "Scrape universe URLs. If not set, load them from univUrls.json and univUrlDict.json")
	parser.add_argument("--storyURLs", action = "store_true", help = "Scrape story URLs. If not set, load them from chapterUrls.json and chapterUrlDict.json. Requires univUrls.json")
	parser.add_argument("--stories", action = "store_true", help = "Scrape story content. Requires chapterUrls.json")
	parser.add_argument("--checkStories", action = "store_true", help = "Scrape story content. Requires the story files created by --stories")
	parser.add_argument("--urlFilter", help = "Scrape only on websites starting with this url (use to filter genres)", default="")
	parser.add_argument("--skipPages", help = "Skip the first N pages when scraping storyURLs (use to continue after failure)", default=0, type=int)
	parser.add_argument("--startAtUniverse", help = "Start at this universe URL when scraping storyURLs (use to continue after failure)")

	args = parser.parse_args()

	log.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
				level=log.DEBUG)
	log.getLogger("requests.packages.urllib3.connectionpool").setLevel(log.INFO)

	if args.universeURLs:
		downloadAndSaveUniverseURLs()

	if args.storyURLs:
		downloadAndSaveStoryURLs(args)

	if args.stories:
		downloadAndSaveStories(args)

	if args.checkStories:
		for __, __, files in os.walk("."):
			totalFiles = len(files)
			for index, file in enumerate(files):
				if index % 10000 == 0:
					log.info("Done with %d files", index)

				if file.startswith("-s-"):
					with open(file) as f:
						content = f.read()
						try:
							json.loads(content)
							continue
						except json.decoder.JSONDecodeError:
							log.warning("File %s is malformatted! ", file)
