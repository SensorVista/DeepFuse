#include "gtest/gtest.h"

#define _CRT_SECURE_NO_WARNINGS

#ifdef __cplusplus
extern "C" {
#endif

#include "dnn/utils/json.h"

/* Existing cJSON implementation code */

// Removed cJSON_GetErrorPtr implementation as it's not needed for tests

// Removed duplicate cJSON_GetStringValue implementation - it already exists in json.cpp

#ifdef __cplusplus
}
#endif

class JsonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test data
        valid_json = R"({
            "name": "Test User",
            "age": 30,
            "active": true,
            "score": 95.5,
            "tags": ["dev", "test"],
            "nested": {
                "field": "value"
            }
        })";

        invalid_json = "{ malformed: json }";
    }

    std::string valid_json;
    std::string invalid_json;
};

// Construction Tests
TEST_F(JsonTest, DefaultConstructsValidEmptyObject) {
    json j;
    EXPECT_NE(j.root(), nullptr);
    // Note: No to_string() method in actual implementation
}

TEST_F(JsonTest, ConstructsFromValidJsonString) {
    json j(valid_json);
    EXPECT_NE(j.root(), nullptr);
}

TEST_F(JsonTest, ThrowsOnInvalidJsonString) {
    EXPECT_THROW(json j(invalid_json), std::runtime_error);
}

// Getter Tests
TEST_F(JsonTest, GetStringReturnsCorrectValue) {
    json j(valid_json);
    std::string result;
    bool success = json::get_string(j.root(), "name", result);
    EXPECT_TRUE(success);
    EXPECT_EQ(result, "Test User");
}

TEST_F(JsonTest, GetStringReturnsFalseForMissingField) {
    json j(valid_json);
    std::string result;
    bool success = json::get_string(j.root(), "nonexistent", result);
    EXPECT_FALSE(success);
}

TEST_F(JsonTest, GetStringReturnsFalseForWrongType) {
    json j(valid_json);
    std::string result;
    bool success = json::get_string(j.root(), "age", result);
    EXPECT_FALSE(success);
}

TEST_F(JsonTest, GetIntReturnsCorrectValue) {
    json j(valid_json);
    int result;
    bool success = json::get_int(j.root(), "age", &result);
    EXPECT_TRUE(success);
    EXPECT_EQ(result, 30);
}

TEST_F(JsonTest, GetBoolReturnsCorrectValue) {
    json j(valid_json);
    bool result;
    bool success = json::get_bool(j.root(), "active", &result);
    EXPECT_TRUE(success);
    EXPECT_EQ(result, true);
}

TEST_F(JsonTest, GetDoubleReturnsCorrectValue) {
    json j(valid_json);
    double result;
    bool success = json::get_double(j.root(), "score", &result);
    EXPECT_TRUE(success);
    EXPECT_DOUBLE_EQ(result, 95.5);
}

TEST_F(JsonTest, GetFloatReturnsCorrectValue) {
    json j(valid_json);
    float result;
    bool success = json::get_float(j.root(), "score", &result);
    EXPECT_TRUE(success);
    EXPECT_FLOAT_EQ(result, 95.5f);
}

// Serialization Tests - Commented out as to_string() method doesn't exist in actual implementation
// TEST_F(JsonTest, ToStringProducesValidJson) {
//     json j(valid_json);
//     std::string serialized = j.to_string();
//     // Should be able to parse the output
//     EXPECT_NO_THROW(json j2(serialized));
// }

// TEST_F(JsonTest, ToStringMaintainsDataIntegrity) {
//     json j(valid_json);
//     std::string serialized = j.to_string();
//     json j2(serialized);
//     
//     EXPECT_EQ(*j.get_string("name"), *j2.get_string("name"));
//     EXPECT_EQ(*j.get_int("age"), *j2.get_int("age"));
// }

// Move Semantics Tests
TEST_F(JsonTest, MoveConstructorTransfersOwnership) {
    json j1(valid_json);
    auto* original_ptr = j1.root();
    
    json j2(std::move(j1));
    EXPECT_EQ(j2.root(), original_ptr);
    EXPECT_EQ(j1.root(), nullptr);
}

TEST_F(JsonTest, MoveAssignmentTransfersOwnership) {
    json j1(valid_json);
    json j2;
    auto* original_ptr = j1.root();
    
    j2 = std::move(j1);
    EXPECT_EQ(j2.root(), original_ptr);
    EXPECT_EQ(j1.root(), nullptr);
}

// Edge Cases
TEST_F(JsonTest, HandlesEmptyStringValues) {
    json j(R"({"empty": ""})");
    std::string result;
    bool success = json::get_string(j.root(), "empty", result);
    EXPECT_TRUE(success);
    EXPECT_EQ(result, "");
}

TEST_F(JsonTest, HandlesNullValues) {
    json j(R"({"null_field": null})");
    std::string str_result;
    int int_result;
    bool str_success = json::get_string(j.root(), "null_field", str_result);
    bool int_success = json::get_int(j.root(), "null_field", &int_result);
    EXPECT_FALSE(str_success);
    EXPECT_FALSE(int_success);
}

TEST_F(JsonTest, HandlesNumericPrecision) {
    json j(R"({"precise": 1234567890.123456789})");
    double result;
    bool success = json::get_double(j.root(), "precise", &result);
    EXPECT_TRUE(success);
    EXPECT_DOUBLE_EQ(result, 1234567890.123456789);
}

// Memory Safety
TEST_F(JsonTest, DoesNotCrashOnMalformedNumeric) {
    json j(R"({"bad_number": 1.2.3})");
    double result;
    bool success = json::get_double(j.root(), "bad_number", &result);
    EXPECT_FALSE(success);
}
