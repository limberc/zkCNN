//
// Generic CLI for zkCNN
//

#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include <iostream>
#include <string>

// the arguments' format
#define INPUT_FILE_ID 1     // the input filename
#define CONFIG_FILE_ID 2    // the config filename
#define OUTPUT_FILE_ID 3    // the input filename
#define PIC_CNT 4           // the number of picture paralleled
#define MODEL_TYPE 5        // the model type (string)
#define NUM_CLASS 6         // the number of classes (optional, default 10)

vector<std::string> output_tb(16, "");

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <config_file> <output_file> <pic_cnt> <model_type> [num_classes]" << std::endl;
        return 1;
    }

    initPairing(mcl::BLS12_381);

    char i_filename[500], c_filename[500], o_filename[500];

    strcpy(i_filename, argv[INPUT_FILE_ID]);
    strcpy(c_filename, argv[CONFIG_FILE_ID]);
    strcpy(o_filename, argv[OUTPUT_FILE_ID]);

    int pic_cnt = atoi(argv[PIC_CNT]);
    std::string model_type = argv[MODEL_TYPE];
    int n_class = 10;
    if (argc > NUM_CLASS) {
        n_class = atoi(argv[NUM_CLASS]);
    }

    output_tb[MO_INFO_OUT_ID] = model_type;
    output_tb[PCNT_OUT_ID] = std::to_string(pic_cnt);

    prover p;
    
    if (model_type == "lenetCifar") {
        // lenetCifar with 32x32, 3 channels.
        lenetCifar nn(32, 32, 3, pic_cnt, MAX, i_filename, c_filename, o_filename, n_class);
        nn.create(p, false);
    } else if (model_type == "lenet") {
        // Standard LeNet (MNIST) 32x32, 1 channel
        lenet nn(32, 32, 1, pic_cnt, MAX, i_filename, c_filename, o_filename);
        nn.create(p, false);
    } else {
        std::cerr << "Unknown model type: " << model_type << std::endl;
        return 1;
    }
    
    verifier v(&p, p.C);
    v.verify();

    for (auto &s: output_tb) printf("%s, ", s.c_str());
    puts("");
}
