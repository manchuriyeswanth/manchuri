#include <iostream>
#include <unordered_set>
#include <queue>
using namespace std;
int main()
{
    //Create 
    unordered_set<int> us={1,22,3,4,4,5};
    // Traverse , any order when sorting is not needed , uses hashing to store so O(1) complexity
    for (auto x:us)
    {
        cout<<""<<x<<endl;
    }
    // Insert using empplace , insert
    us.insert(10);
    auto r=us.emplace(20);
    cout<<""<<*(r.first)<<endl;

    cout<<""<<r.second<<endl;
    for (auto x:us)
    {
        cout<<""<<x<<endl;
    }
    //Priority Queue

    priority_queue<int> pq;
    pq.push(13);
    pq.push(12);
    pq.push(100);
    while(!pq.empty())
    {
        cout<<""<<pq.top()<<endl;
        pq.pop();
    }
    //Priority Queue Declaration priority_queue<int, vector<int>, greater<int>> pq; /data type -> underlying STL Contrainer -> PRedicate Function for minheap
    priority_queue<int, vector<int>, greater<int>> pq_min;
    pq_min.push(99);
    pq_min.push(24);
    pq_min.push(66);
    pq_min.push(13);
    while(!pq_min.empty())
    {
        cout<<""<<pq_min.top()<<endl;
        pq_min.pop();
    }
    return 0;

}