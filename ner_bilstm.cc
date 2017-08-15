//
// Created by jeffly on 17-8-10.
//
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


using namespace std;
using namespace tensorflow;

int main()
{
    const string pathToGraph = "ner_model_cc_debug/ner_bilstm.ckpt.meta";
    const string checkpointPath = "ner_model_cc_debug/ner_bilstm.ckpt";
    Status status;
    Session* session;
    status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        throw runtime_error("Could not create Tensorflow session.");
    }


    std::string order;
// Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

// Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

// Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }
    cout<<"model load done"<<endl;

// 构造输入debug
    std::vector<std::pair<string, Tensor>> feed_inputs;
    const string x_name = "seg_var_scope/input_data";
    const string y_name = "seg_var_scope/targets";
    const string len_name = "seg_var_scope/seq_len";
    const string seg_name = "seg_var_scope/seg_data";
    const string max_len_name = "seg_var_scope/max_seq_len";
    const string dropout_name = "seg_var_scope/Dropout";

    const string decode_name = "seg_var_scope/ReverseSequence_1";


    int i,j;

    int node_count = graph_def.graph_def().node_size();
    for (i = 0; i < node_count; i++)
    {
        auto n = graph_def.graph_def().node(i);

        if (n.name()== decode_name)
        {
            cout<<"find decode name"<<endl;
        }
    }
    int batch_len;
    cout<<"batch len:";
    cin>>batch_len;
//    x data
    Tensor x(tensorflow::DT_INT32,{20,batch_len});
    auto x_map = x.tensor<int, 2>();
    for (i=0;i<20;i++)
    {
        for (j=0;j<batch_len;j++)
        {
            x_map(i,j) = 50;
        }
    }
    feed_inputs.emplace_back(x_name, x);
// y data
    Tensor y(tensorflow::DT_INT32,{20,batch_len});
    auto y_map = y.tensor<int, 2>();
    for (i=0;i<20;i++)
    {
        for (j=0;j<batch_len;j++)
        {
            y_map(i,j) = 5;
        }
    }
    feed_inputs.emplace_back(y_name, y);
//    len data
    Tensor len(tensorflow::DT_INT32, {20});
    auto len_map = len.tensor<int, 1>();
    for(i=0;i<20;i++)
    {
        len_map(i) = batch_len-i;
    }
    feed_inputs.emplace_back(len_name, len);
//    seg data
    Tensor seg(tensorflow::DT_INT32,{20,batch_len});
    auto seg_map = seg.tensor<int, 2>();
    for (i=0;i<20;i++)
    {
        for (j=0;j<batch_len;j++)
        {
            seg_map(i,j) = 1;
        }
    }
    feed_inputs.emplace_back(seg_name, seg);
//    max len
    Tensor max_len(tensorflow::DT_INT32, {1});
    auto max_len_map = max_len.tensor<int, 1>();
    max_len_map(0) = batch_len;
    feed_inputs.emplace_back(max_len_name, max_len);

//    dropout
    Tensor dropout(tensorflow::DT_FLOAT, {1});
    auto dropout_map = dropout.tensor<float, 1>();
    dropout_map(0) = 1.0;
    feed_inputs.emplace_back(dropout_name, dropout);

    std::cout<<"input done"<<endl;
//  sess run
    std::vector<tensorflow::Tensor> fetch_outputs;
    std::vector<std::string> output_names;
    output_names.emplace_back(decode_name);
//    ClientSession
    cout<<"x data:"<<x.DebugString()<<endl;
    cout<<"y data:"<<y.DebugString()<<endl;
    cout<<"len data:"<<len.DebugString()<<endl;
    cout<<"seg data:"<<seg.DebugString()<<endl;
    cout<<"max len data:"<<max_len.DebugString()<<endl;
    cout<<"dropout:"<<dropout.DebugString()<<endl;
    status = session->Run(feed_inputs, {decode_name}, {}, &fetch_outputs);

    if (!status.ok()) {
        cout<<"run error"<<endl;
        cout<<status.ToString()<<endl;
        return 0;
    }
    cout<<"run over"<<endl;
//    output
    cout<<fetch_outputs.size()<<endl;
    cin>>order;
    cout<< fetch_outputs[0].DebugString()<<endl;

    cin>>order;
    auto crf_decode_map = fetch_outputs[0].tensor<int,2>();
    for(i=0;i<20;i++)
    {
        for (j = 0; j < 103; j++)
        {
            std::cout<<crf_decode_map(i,j)<<" ";
        }
        cout<<endl;
    }
    cout<<"finished"<<endl;

    return 0;
}
