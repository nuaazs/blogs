被一个很简单的问题折磨了一个小时，有必要记录一下。
首先是直接找了当时再quicker上面用的Mysql.data.dll文件在Visual Studio里面添加引用然后查询数据库。
```csharp
using System;
using MySql.Data.MySqlClient;
using System.Collections.Generic;
using System.IO;


namespace ConsoleApp4
{
	class Program
	{
		static void Main(string[] args)
		{
			//Console.WriteLine("Hello World!");

			List<string> name_list = new List<string>();
			List<string> send_num = new List<string>();
			List<string> back_num = new List<string>();

			String connetStr = "server=xxxxxxx;port=xxxx;user=xxxx;password=xxxxx;database=xxxxxxx;";

			MySqlConnection conn = new MySqlConnection(connetStr);
			conn.Open();
			try
			{

				string sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() GROUP BY owner ORDER BY COUNT desc;";
				MySqlCommand cmd = new MySqlCommand(sqlStr, conn);

				MySqlDataReader rdr = cmd.ExecuteReader();
				try
				{
					while (rdr.Read())
					{
						name_list.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());
					}

				}
				catch (Exception ex)
				{
					Console.WriteLine(ex.ToString());
				}


				string temp_sqlStr = "";

				foreach (string name in name_list)
				{
					temp_sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() and `owner` = '" + name + "' LIMIT 1;";
					MySqlCommand cmd1 = new MySqlCommand(sqlStr, conn);
					MySqlDataReader rdr1 = cmd1.ExecuteReader();
					while (rdr.Read())
					{
						send_num.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());
					}

					temp_sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() and `owner` = '" + name + "' LIMIT 1;";
					MySqlCommand cmd2 = new MySqlCommand(sqlStr, conn);
					MySqlDataReader rdr2 = cmd2.ExecuteReader();
					while (rdr2.Read())
					{
						back_num.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());
					}
				}
			}
			catch (MySqlException ex)
			{
				Console.WriteLine(ex.Message);
			}
			finally
			{
				conn.Close();
			}
			Console.ReadLine();

		}
	}
}


```
报如下的错误：
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BqPAd.jpg)
原因：Mysql.Data.dll文件版本不对，和.Net框架版本对不上。
解决方法：下载对应版本的文件或者使用nuget
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/jhLv3.jpg)


然后报了另一个错误如下图：
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/kqx3k.jpg)
在(这里)[https://stackoverflow.com/questions/46542810/connect-to-mysql-using-ssl-in-c-sharp?rq=1]找到了答案,添加 `SSL MODE = 0`。
```csharp
using System;
using MySql.Data.MySqlClient;
using System.Collections.Generic;
using System.IO;


namespace ConsoleApp4
{
	class Program
	{
		static void Main(string[] args)
		{
			//Console.WriteLine("Hello World!");

			List<string> name_list = new List<string>();
			List<string> send_num = new List<string>();
			List<string> back_num = new List<string>();



			String connetStr = "server=xxxx;port=xxxx;user=zhaosheng;password=xxxx;database=buxijiao;SSL Mode=0";

			MySqlConnection conn = new MySqlConnection(connetStr);

			conn.Open();

			try
			{



				string sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() GROUP BY owner ORDER BY COUNT desc;";


				MySqlCommand cmd = new MySqlCommand(sqlStr, conn);

				MySqlDataReader rdr = cmd.ExecuteReader();


				try
				{
					while (rdr.Read())
					{
						name_list.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());


					}

				}


				catch (Exception ex)
				{
					Console.WriteLine(ex.ToString());
				}


				string temp_sqlStr = "";

				foreach (string name in name_list)
				{
					temp_sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() and `owner` = '" + name + "' LIMIT 1;";
					MySqlCommand cmd1 = new MySqlCommand(sqlStr, conn);
					MySqlDataReader rdr1 = cmd1.ExecuteReader();
					while (rdr.Read())
					{
						send_num.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());
					}

					temp_sqlStr = "select owner ,count(*) AS COUNT from ysy_log_v2 where `type`='+' and `date`= curdate() and `owner` = '" + name + "' LIMIT 1;";
					MySqlCommand cmd2 = new MySqlCommand(sqlStr, conn);
					MySqlDataReader rdr2 = cmd2.ExecuteReader();
					while (rdr2.Read())
					{
						back_num.Add(rdr[0].ToString());
						Console.WriteLine(rdr[0].ToString());
					}

				}

			}


			catch (MySqlException ex)
			{
				Console.WriteLine(ex.Message);
			}
			finally
			{
				conn.Close();
			}
			Console.ReadLine();

		}
	}
}


```