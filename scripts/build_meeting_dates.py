import pandas as pd
import cloudscraper
import re
from datetime import datetime
from bs4 import BeautifulSoup


def get_fomc_dates():

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    scraper = cloudscraper.create_scraper(
        interpreter='nodejs',  # Use Node.js instead of native solvers for tougher challenges
        browser={
            'browser': 'chrome',
            'platform': 'linux',
            'desktop': True
        })

    html = scraper.get(url='https://en.wikipedia.org/wiki/History_of_Federal_Open_Market_Committee_actions', headers=headers) # Scrape the wikipedia page for Historical FOMC Actions
    html_content = html.text # Extract the text from the website


    '''
    Match on format:
    (<td>)(Capital Letter)(One or more lowercase letters)(Space)(One or two digits)(Comma)(Space)(Four Digits)(Zero or more spaces)(</td>)
    '''
    pattern = r"<td>[A-Z][a-z]+\s\d{1,2},\s\d{4}\s*\n</td>"
    dates = re.findall(pattern, html_content)

    '''
    Strip the <td> / </td> labels and spaces in the text, I pattern match using them as
    I want to make sure to only pull the dates that were in the table.

    Then we convert the text to datetime format with strptime,
    and turn it back to our desired format with strftime.
    '''

    dates = [datetime.strptime(date.strip("<td>/ "),"%B %d, %Y ").strftime("%Y-%m-%d")for date in dates]

    dates_df = pd.DataFrame({"FOMC Meeting Dates": sorted(set(dates))}) # Turn sorted unique dates into a df

    return dates_df