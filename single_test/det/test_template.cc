
template <typename T1, typename T2> void fun(T1, T2) {}

template <typename T1, typename T2> void fun(T2, T1) {}

int main() { fun(1, 2.0f); }