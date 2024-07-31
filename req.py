"""
spider test
"""

import requests

url = 'https://fd.aigei.com/src/vdo/mp4/cd/cd02a577070a474fb62dfb8f48feab22.mp4'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

re = requests.get(url=url, headers=headers)


print(re)