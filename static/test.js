
// Button Listener
document.querySelector("#postBtn").onclick = function(){
    console.log("Running");
    run();
};

function run(){

    let url = "http://localhost:8080/setModel/";
    let modelStruct = {
        type:                       "sequential",
        input_shape:                [1, 2], 
        optimizer:                  'adam',
        loss:                       'categorical_crossentropy',
        metrics:  [
                                    'accuracy',
        ],
        layers:
            [
                {
                    type:           "dense",
                    nodes:          1,
                    activation:     'relu',
                },
                {
                    type:           'dense',
                    nodes:          3,
                    activation:     'relu',
                },
                {
                    type:           'dropout',
                    keep_prob:      0.4
                },
                {
                    type:           "dense",
                    nodes:          1,
                    activation:     'softmax',
                }
            ]
    }

    sendJSON(modelStruct, url)


}

function sendJSON(struct, url){ 

    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {

            console.log("Recieved!")
            var json = JSON.parse(xhr.responseText);
            responseHandler(json)


        }
    };
    var data = JSON.stringify(struct);
    xhr.send(data);
}

function responseHandler(json){
    document.querySelector("body").innerHTML += JSON.stringify(json);
}
