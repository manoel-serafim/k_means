INC_DIR := ../include
SRC_DIR := ../source
BUILD_DIR := .
HEAD_DIR := ..
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj
TEST_DIR := ../test


SRCS := $(shell find $(SRC_DIR) -name "*.c")
HEADERS := $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*/*.h)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

CC := clang
CFLAGS := -Wall -Wextra -Werror -pedantic -O3 -I$(INC_DIR) \
          -Waddress -Warray-bounds -Wcast-align -Wconversion \
          -Wfloat-equal -Wformat -Wimplicit-function-declaration \
          -Wmissing-prototypes -Wnull-dereference -Wshadow \
          -Wsign-compare -Wstrict-aliasing -Wuninitialized \
          -Wunused-variable

TARGET := $(BIN_DIR)/sequential
LDFLAGS := -lm

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

CRUST := $(patsubst $(SRC_DIR)/%.c, $(SRC_DIR)/%.uncrustify, $(SRCS))
CRUST += $(patsubst $(INC_DIR)/%.h, $(INC_DIR)/%.uncrustify, $(HEADERS))


uncrustify: $(CRUST)

../%.uncrustify: ../%.c
	uncrustify -c ../.uncrustify.cfg -f $< > $@

../%.uncrustify: ../%.h
	uncrustify -c ../.uncrustify.cfg -f $< > $@

cppcheck:
	cppcheck --force -q --check-level=exhaustive --enable=all \
		$(SRCS) $(HEADERS) -I $(INC_DIR) \
		--checkers-report=cppcheck_report.xml

test: $(TARGET)
	@echo "Running K-means tests..."
	@mkdir -p $(TEST_DIR)/results
	@for point_file in $(TEST_DIR)/point/*; do \
		point_name=$(basename $point_file .txt); \
		for centroid_file in $(TEST_DIR)/centroid/*; do \
			centroid_name=$(basename $centroid_file .txt); \
			result_dir=$(TEST_DIR)/results/${point_name}_${centroid_name}; \
			mkdir -p $result_dir; \
			echo "Test ${point_name}_${centroid_name}: points=$point_file centroids=$centroid_file"; \
			$(TARGET) $point_file \
			          $centroid_file \
			          50 \
			          0.000001 \
			          $result_dir/assign.txt \
			          $result_dir/centroids.txt; \
		done; \
	done
	@echo "All tests completed!"

.PHONY: all clean uncrustify cppcheck test
