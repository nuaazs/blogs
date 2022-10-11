通过C++进行单链表的创建、打印以及使用栈实现逆序打印。

单链表的创建和打印

```cpp
#include <iostream>
using namespace std;
//定义结构体
struct ListNode{
    int val;
    ListNode* next;
}

class Solution
{
    //创建单链表
    void createList(ListNode *head){
        int i;
        ListNode* phead = head;//不破坏头指针
        for(i=1;i<10;i++){
            ListNode* node = new ListNode;
            node->val=i;
            node->next=NULL;
            phead->next=node;
            phead=node;
        }
        cout<<"链表创建成功！"<< endl;
    }
    //打印链表
    void printList(ListNode* head){
        ListNode* phead=head;
        while(phead!=NULL){
            cout<<phead->val<<" ";
            phead=phead->next;
        }
        cout<<endl;
    }
};
int main(){
    ListNode* head = new ListNode;
    Solution l1;
    head->val=0;
    head->next=NULL;
    l1.createList(head);
    l1.printList(head);
    return 0;
}
```

 逆序打印单链表的方式

```cpp
#include <iostream>
#include <vector>
#include <stack>
using namespace std;
//定义结构体
struct ListNode{
    int val;
    ListNode* next;
};

class Solution{
    public:
    //创建单链表
    void createList(ListNode *head){
        int i;
        ListNode* phead=head;
        for(i=1;i<10;i++){
            ListNode* phead=head;//不破坏头指针
            for(i=1;i<10;i++){
                ListNode* node = new ListNode;
                node->val=i;
                node->next=NULL;
                phead->next=node;
                phead=node;
            }
            cout<<"链表创建成功！"<< endl;
        }
        // 打印链表
        void printList(ListNode* head){
            ListNode* phead=head;
            while(phead!=NULL){
                cout<<phead->val<<" ";
                phead=phead->next;
            }
            cout << "\n";
        }
        //利用栈先进后出的思想
        vector<int> printListInverseByStack(ListNode* head){
            vector<int> result;
            stack<int> arr;
            ListNode* phead=head;
            whild(phead!=NULL){
                arr.push(phead->val);
                phead=phead->next;
            }
            whild(!arr.empty()){
                result.push_back(arr.top());
                arr.pop();
            }            
            return result;
        }
        void printVector(vector<int> result){
            int i;
            for(i=0;i<result.size();i++){
                cout<<result[i]<<" ";
            }
            cout<<"\n";
        }
    }
};

int main(){
    ListNode* head = new ListNode;
    vector<int> result;
    Solution l1;
    head->val=0;
    head->next=NULL;
    l1.createList(head);
    l1.printList(head);
    //利用栈逆序
    result=l1.printListInverseByStack(head);
    cout<<"利用栈逆序的结果为："<< endl;
    l1.printVector(result);
    return 0;
}
```





单链表，弄清楚和`stl`中`list`的区别

ListNode的结构

```cpp
struct ListNode{
    int val; //当前结点的值
    ListNode *next; //指向下一个结点的指针
    ListNode(int x) :val(x),next(NULL){} //初始化当前结点值为x，指针为空
}
```

如何向ListNode中插入新的结点：

从键盘输入

```cpp
ListNode* temp1 = new Solution::ListNode(0); //创建新元素
ListNode* listnode1 = temp1; //最后的结果指向temp1，这样可以获取temp所接收的全部元素，而temp的指针由于每次都往下移所以每次都更新

while((c=getchar()) != '\n') //以空格区分各个结点的值
{
    if (c != ' '){
        ungetc(c, stdin);//把不是空格的字符丢回去
        cin >> num;
        Solution::ListNode* newnode = new Solution::ListNode(0);
        newnode->val = num; //创建新的结点存放键盘中读入的值
        newnode->next = NULL;
        temp2->next=newnode; //并将其赋值给temp2
        temp2 = newnode;//此处也可以写成  temp2=temp2->next,使指针指向下一个，以待接收新元素
    }
}
```

逆序输出所有元素

```cpp
void  Solution::reversePrintListNode(ListNode* head)
{
    if (head == nullptr) return;
     cout << head->val; //顺序输出
    reversePrintListNode(head->next);  
    cout << head->val; //逆序输出
   
}
```



