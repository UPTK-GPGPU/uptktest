#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>

int main(int argc, char * argv[])
{
    //testing::GTEST_FLAG(device_name) = "Vega 20";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
