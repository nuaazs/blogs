更换腾讯云服务器。

1. 实力迁移

2. ssh新实例，通过conf.php查看wordpress的mysql密码

3. 运行以下SQL代码：

   ```sql
   UPDATE wp_options SET option_value = replace(option_value, '旧网址','新网址') ;    
   UPDATE wp_posts SET post_content = replace(post_content, '旧网址','新网址') ;    
   UPDATE wp_comments SET comment_content = replace(comment_content, '旧网址', '新网址') ;    
   UPDATE wp_comments SET comment_author_url = replace(comment_author_url, '旧网址', '新网址') ;
   ```

   

