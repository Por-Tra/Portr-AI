# Avec les paramètres de Crawler/config.py
python crawl.py

# Override en ligne de commande
python crawl.py --url https://example.com --max-pages 200 --max-depth 4 --workers 10

# Aide
python crawl.py --help

# Plusieurs sites
# Plusieurs URLs en ligne de commande
python crawl.py --url https://site1.com https://site2.com https://site3.com

# Avec limite par site
python crawl.py --max-pages-per-site 500 --max-pages 5000



# ========================================================================================
# Liste d'agent HTTP pour le crawler
# ========================================================================================
Chrome sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36
Chrome sur macOS : Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36
Firefox sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0
Firefox sur Linux : Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0
Safari sur macOS : Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15
Edge sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58
Chrome sur Android : Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36
Safari sur iOS : Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1



# ========================================================================================
# Liste de sites web pour tester le crawler
# ========================================================================================
books.toscrape.com (simule une librairie en ligne avec livres et prix)
quotes.toscrape.com (citations et auteurs, idéal pour extraire du texte structuré)
scrapethissite.com (divers défis comme des tableaux de hockey ou de films)
oxylabs.io/sandbox (sandbox gratuit pour tester le scraping avec des données réelles)
httpbin.org (pour tester des requêtes HTTP et responses)
crawler-test.com (site de test pour crawlers, avec divers scénarios)
jsonplaceholder.typicode.com (API JSON fake pour pratique)
mockaroo.com (générateur de données mock pour scraping)
the-internet.herokuapp.com (tests pour formulaires, authentification, etc.)
webscraper.io/test-sites (pages de test avec tableaux, e-commerce, etc.)

# Sites financiers (données publiques sur les marchés) :

finance.yahoo.com (Yahoo Finance : cours boursiers, historiques ; tolère le scraping modéré)
google.com/finance (Google Finance : données similaires, souvent scrapé pour analyses)
data.worldbank.org (World Bank Open Data : statistiques économiques mondiales)

# Sites sportifs (stats et résultats) :

espn.com (ESPN : scores, stats joueurs ; vérifiez les sections publiques)
sports-reference.com (Sports Reference : données historiques sur divers sports)

# Sites météo (données ouvertes) :

openweathermap.org (OpenWeather : APIs et données météo publiques)
wunderground.com (Weather Underground : historique météo, tolère le scraping léger)

# Bases de données publiques et gouvernementales :

data.gov (données ouvertes du gouvernement US : santé, éducation, etc.)
data.europa.eu (données de l'UE : similaires, gratuites et ouvertes)

# Sites d'emploi et annuaires :

indeed.com (offres d'emploi ; scraping autorisé pour usage personnel, mais limité)
glassdoor.com (salaires, avis ; vérifiez les termes)
yellowpages.com (annuaires locaux ; souvent scrapé pour contacts)
yelp.com (avis restaurants ; tolère le scraping modéré via API publique)

# Sites de voyage et hôtellerie :

tripadvisor.com (avis hôtels, restaurants ; beaucoup scrapé, mais respectez les limites)
booking.com (prix hôtels ; autorise via API ou scraping éthique)
skyscanner.net (vols et prix ; données publiques pour comparaison)

# Autres sites populaires tolérants (basé sur robots.txt allowing all ou usage courant) :

wikipedia.org (articles encyclopédiques ; autorise le scraping via API ou directement, mais pas pour usage commercial massif)
ebay.com (listings produits ; scraping pour prix, mais utilisez l'API de préférence)
twitter.com (ou x.com : posts publics ; autorise via API, scraping limité)
360.cn (site chinois avec robots.txt allowing all)
fc2.com (blogging japonais, allowing all)
livejasmin.com (allowing all via robots.txt)
cnzz.com (pas de robots.txt, donc potentiellement ouvert)
diply.com (pas de robots.txt)

# Site de media comme bfmtv.com, cnews, ...
# ========================================================================================


# meilleur site :
http://mumm.hautetfort.com/tres_long_texte/

