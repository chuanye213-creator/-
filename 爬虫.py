import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

# ===================== 东北财经大学专用配置 =====================
BASE_DOMAIN = "https://www.dufe.edu.cn"
BASE_URL = "https://www.dufe.edu.cn/news/news/{page}.html"
START_PAGE = 1
END_PAGE = 4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.dufe.edu.cn/",
}
DELAY = 1
# ====================================================================

def crawl_one_page(page):
    news_list = []
    url = BASE_URL.format(page=page)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")

        # 找到所有新闻（放宽匹配条件）
        news_items = soup.find_all("a", href=lambda x: x and "content_" in str(x))

        for item in news_items:
            # 修复：多规则提取标题（覆盖所有情况）
            title = ""
            # 规则1：找包含title的div（不管class名）
            title_div = item.find("div", string=lambda s: s and len(s.strip())>2)
            if title_div:
                title = title_div.get_text(strip=True)
            # 规则2：直接取a标签里的所有文本（兜底）
            if not title:
                title = item.get_text(strip=True).replace("\n","").replace(" ","")
            
            # 日期提取（同样兜底）
            date = ""
            time_div = item.find("div", string=lambda s: s and "." in str(s) and len(s.strip())==10)
            if time_div:
                date = time_div.get_text(strip=True)

            # 链接
            href = item.get("href", "")
            full_link = f"{BASE_DOMAIN}/news/news/{href}" if href else ""

            # 过滤空标题（避免无意义数据）
            if title and len(title) > 2:
                news_list.append({
                    "标题": title,
                    "日期": date,
                    "链接": full_link,
                    "来源页": f"第{page}页"
                })

        print(f"✅ 第{page}页爬取完成，本次获取 {len(news_list)} 条有效新闻")

    except Exception as e:
        print(f"❌ 第{page}页爬取失败：{str(e)}")

    time.sleep(DELAY)
    return news_list

def main():
    print("=" * 50)
    print("      东北财经大学新闻爬虫开始运行")
    print("=" * 50)

    all_news = []
    for page in range(START_PAGE, END_PAGE + 1):
        page_news = crawl_one_page(page)
        all_news.extend(page_news)

    # 去重 + 保存
    df = pd.DataFrame(all_news)
    df = df.drop_duplicates(subset=["标题"], keep="first")
    df.to_csv("school_news.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 50)
    print(f"🎉 爬取结束！总共获取有效新闻：{len(df)} 条")
    print(f"📄 数据已保存到：school_news.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()