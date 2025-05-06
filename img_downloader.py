from icrawler.builtin import GoogleImageCrawler
from icrawler import ImageDownloader

class FilteredDownloader(ImageDownloader):
    def download(self, task, default_ext, timeout=5, **kwargs):
        super().download(task, default_ext, timeout, **kwargs)
        from PIL import Image
        img = Image.open(task['file_path'])
        w, h = img.size
        if not (512 <= w <= 1080 and 512 <= h <= 1080):
            import os; os.remove(task['file_path'])

queries = [
    "dog"
]

crawler = GoogleImageCrawler(storage={'root_dir': 'dog'})
crawler.crawl(keyword=queries[0],
              max_num=1000,
              file_idx_offset=0,
              overwrite=False)
