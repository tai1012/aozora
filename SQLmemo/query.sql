select title from text
	group by author, url, ;  -- 16866件取得　著作権あり含んでしまっている

select author, count(id) from text
group by author
having count(id) > 50 
order by count(id) desc;

/*
distinct 著作権ありはぶく他の方法も考える
作者の書籍50件以下除外
大久保 ゆう この人だけ除外 そもそも入ってないので、無視
目視で確認したところ、著作権に該当するものはないので上のSQLで出力された人物を使いたい。
*/

--　念の為タイトルがあってるかの確認
select get_title, title from text
where get_title != title;
-- 旧漢字で異なるものやアルファベットで異なるものが存在することを確認できた。
-- 現段階ではtextページの前の詳細ページに書いてあるタイトルを採用することにする。
-- 旧漢字が扱いにくいため

select author, count(author) from text
group by author
having count(id) > 50 
order by count(id) desc;
-- 今回使いたい著者の中にはいないことを確認。

-- 取得したい人の中で作品のURLが被っているものはないか確認
select get_title as title, author, url, text from text
	where author in (select author from text
						group by author
						having count(id) > 50)
	group by author, get_title, url, text
	order by author;
-- 8990件の著作権が切れている書籍の取得に成功

-- 確認
select author, count(author) from text
	where author in (select author from text
						group by author
						having count(id) > 50)
	group by author
	order by count(author) desc;