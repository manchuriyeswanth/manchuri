#include <iostream>
#include <set>
using namespace std;
int main()
{
    set<int> s1;
    set<int> s2={1,2,3,4,2,1,3};
    for (auto& x:s2)
    {
        cout<<""<<x<<endl;
    }
    cout<<endl;
    return 0;
}