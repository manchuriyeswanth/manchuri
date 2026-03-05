#include <iostream>
#include <unordered_set>
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
}