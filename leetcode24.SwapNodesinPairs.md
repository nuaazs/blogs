## 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head.

 

**Example 1:**

![image-20210309165915103](C:\Users\nuaazs\AppData\Roaming\Typora\typora-user-images\image-20210309165915103.png)

```cpp
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```

**Example 2:**

```cpp
Input: head = []
Output: []
```

**Example 3:**

```cpp
Input: head = [1]
Output: [1]
```

 

**Constraints:**

- The number of nodes in the list is in the range `[0, 100]`.
- `0 <= Node.val <= 100`



### Solution

Pointer-pointer `pp` points to the pointer to the current node. So at first, `pp` points to `head`, and later it points to the `next` field of ListNodes. Additionally, for convenience and clarity, pointers `a` and `b` point to the current node and the next node.

#### cpp

We need to go from `*pp == a -> b -> (b->next)` to `*pp == b -> a -> (b->next)`. The first three lines inside the loop do that, setting those three pointers (from right to left). The fourth line moves `pp` to the next pair.

```cpp
ListNode* swapPairs(ListNode* head){
	ListNode **p = &head, *a, *b;
	while((a=*pp) && (b=a->next)){
		a->next = b->next;
		b->next = a;
		*pp = b;
		pp = &(a->next);
	}
	return head;
}
```



```cpp
class Solution{
	public:
	ListNode* swapPairs(ListNode* head){
        if(head == NULL)
            return NULL;
        if(head->next == NULL)
            return head;
        ListNode* next = head->next;
        next->next = head;
        return next;
    }
};
```

```cpp
class Solution{
    public:
    //invert the first 2 and then recurse for the rest
    ListNode* swapPairs(ListNode* head){
        //base case here
        if(!head || !head->next) return head;
        //Swapping part happens here, please draw to visualize
        ListNode *temp = head->next;
        head->next = swapPairs(temp->next);
        temp->next = head;
        return temp;
    }
};
```

 it may be more clear to others by introducing a second tmp variable.

Comments are for the input list [1, 2, 3]

```cpp
ListNode* swapPairs(ListNode* head){
    if(head == nullptr || head->next == nullptr) return head;
    // 2 is new head, 1 is head
    ListNode* new_head = head->next;
    // store 3
    ListNode* third = head->next->next;
    
    //2->1
    new_head->next = head;
    //1->3
    head->next = swapPairs(third);
    return new_head;
}
```







#### python

Here, `pre` is the previous node. Since the head doesn't have a previous node, I just use `self` instead. Again, `a` is the current node and `b` is the next node.



To go from `pre -> a -> b -> b.next` to `pre -> b -> a -> b.next`, we need to change those three references. Instead of thinking about in what order I change them, I just change all three at once.

```python
def swapParis(self, head):
    pre, pre.next = self, head
    while pre.next and pre.next.next:
        a = pre.next
        b = a.next
        pre.next, b.next, a.next = b, a, b.next
        pre = a
    return self.next
```
