#include "gtest/gtest.h"

#define _CRT_SECURE_NO_WARNINGS

#ifdef __cplusplus
extern "C" {
#endif

#include "json.h"

/* Existing cJSON implementation code */

const char* cJSON_GetErrorPtr(void)
{
    return (const char*)(global_error.json + global_error.position);
}

char* cJSON_GetStringValue(const cJSON* const item) {
    if (!cJSON_IsString(item)) {
        return NULL;
    }
    return item->valuestring;
}

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
    EXPECT_EQ(j.to_string(), "{}");
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
    auto result = j.get_string("name");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Test User");
}

TEST_F(JsonTest, GetStringReturnsNulloptForMissingField) {
    json j(valid_json);
    auto result = j.get_string("nonexistent");
    EXPECT_FALSE(result.has_value());
}

TEST_F(JsonTest, GetStringReturnsNulloptForWrongType) {
    json j(valid_json);
    auto result = j.get_string("age");
    EXPECT_FALSE(result.has_value());
}

TEST_F(JsonTest, GetIntReturnsCorrectValue) {
    json j(valid_json);
    auto result = j.get_int("age");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 30);
}

TEST_F(JsonTest, GetBoolReturnsCorrectValue) {
    json j(valid_json);
    auto result = j.get_bool("active");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, true);
}

TEST_F(JsonTest, GetDoubleReturnsCorrectValue) {
    json j(valid_json);
    auto result = j.get_double("score");
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 95.5);
}

TEST_F(JsonTest, GetFloatReturnsCorrectValue) {
    json j(valid_json);
    auto result = j.get_float("score");
    EXPECT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(*result, 95.5f);
}

// Serialization Tests
TEST_F(JsonTest, ToStringProducesValidJson) {
    json j(valid_json);
    std::string serialized = j.to_string();
    // Should be able to parse the output
    EXPECT_NO_THROW(json j2(serialized));
}

TEST_F(JsonTest, ToStringMaintainsDataIntegrity) {
    json j(valid_json);
    std::string serialized = j.to_string();
    json j2(serialized);
    
    EXPECT_EQ(*j.get_string("name"), *j2.get_string("name"));
    EXPECT_EQ(*j.get_int("age"), *j2.get_int("age"));
}

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
    auto result = j.get_string("empty");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, "");
}

TEST_F(JsonTest, HandlesNullValues) {
    json j(R"({"null_field": null})");
    EXPECT_FALSE(j.get_string("null_field").has_value());
    EXPECT_FALSE(j.get_int("null_field").has_value());
}

TEST_F(JsonTest, HandlesNumericPrecision) {
    json j(R"({"precise": 1234567890.123456789})");
    auto result = j.get_double("precise");
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 1234567890.123456789);
}

// Memory Safety
TEST_F(JsonTest, DoesNotCrashOnMalformedNumeric) {
    json j(R"({"bad_number": 1.2.3})");
    auto result = j.get_double("bad_number");
    EXPECT_FALSE(result.has_value());
}
