# Avec les paramètres de Crawler/config.py
```bash
python crawl.py
```
# Override en ligne de commande
```bash
python crawl.py --url https://example.com --max-pages 200 --max-depth 4 --workers 10
```


# Aide
```bash
python crawl.py --help
```


# Plusieurs URLs en ligne de commande
```bash
python crawl.py --url https://site1.com https://site2.com https://site3.com
```

# Avec limite par site
```bash
python crawl.py --max-pages-per-site 500 --max-pages 5000
```


# Liste d'agent HTTP pour le crawler
```bash
Chrome sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36
Chrome sur macOS : Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36
Firefox sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0
Firefox sur Linux : Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0
Safari sur macOS : Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15
Edge sur Windows : Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58
Chrome sur Android : Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36
Safari sur iOS : Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1
```



