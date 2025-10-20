INC_DIR := include
SRC_DIR := source
BUILD_DIR := build
HEAD_DIR := .
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj

SRCS_SEQ := $(filter-out $(SRC_DIR)/openmp.c, $(shell find $(SRC_DIR) -name "*.c"))
SRCS_OMP := $(SRC_DIR)/openmp.c
HEADERS := $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*/*.h)
OBJS_SEQ := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/seq/%.o, $(SRCS_SEQ))
OBJS_OMP := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/omp/%.o, $(SRCS_OMP))

CC := clang
CFLAGS := -Wall -Wextra -Werror -pedantic -O3 -I$(INC_DIR) \
          -Waddress -Warray-bounds -Wcast-align -Wconversion \
          -Wfloat-equal -Wformat -Wimplicit-function-declaration \
          -Wmissing-prototypes -Wnull-dereference -Wshadow \
          -Wsign-compare -Wstrict-aliasing -Wuninitialized \
          -Wunused-variable -fopenmp

TARGET_SEQ := $(BIN_DIR)/sequential
TARGET_OMP := $(BIN_DIR)/openmp
LDFLAGS := -lm -fopenmp

all: $(TARGET_SEQ) $(TARGET_OMP)

$(TARGET_SEQ): $(OBJS_SEQ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TARGET_OMP): $(OBJS_OMP)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/seq/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/omp/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

CRUST := $(patsubst $(SRC_DIR)/%.c, $(SRC_DIR)/%.uncrustify, $(SRCS_SEQ) $(SRCS_OMP))
CRUST += $(patsubst $(INC_DIR)/%.h, $(INC_DIR)/%.uncrustify, $(HEADERS))

uncrustify: $(CRUST)

$(SRC_DIR)/%.uncrustify: $(SRC_DIR)/%.c
	uncrustify -c .uncrustify.cfg -f $< -o $@

$(INC_DIR)/%.uncrustify: $(INC_DIR)/%.h
	uncrustify -c .uncrustify.cfg -f $< -o $@

cppcheck:
	cppcheck --force -q --check-level=exhaustive --enable=all \
		$(SRCS_SEQ) $(SRCS_OMP) $(HEADERS) -I $(INC_DIR) \
		--checkers-report=cppcheck_report.xml

.PHONY: all clean uncrustify cppcheck