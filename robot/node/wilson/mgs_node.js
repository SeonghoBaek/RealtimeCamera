var nbus = require('./nodebus');
var db = require('./mgs');

var MGS_NODE_NAME = "mgs.node";

// syntax: collection$json
var save = function(document) {
    //console.log(document.toString());

    var docString = document.toString();

    var query = docString.split("$");

    var collection = query[0];

    if (collection == "log") {

        var queryString = query[1].replace(/\n/g, "\\n");

        var jsonNew = null;

        try {
            jsonNew = JSON.parse(queryString);
        } catch (e) {
            console.log("Syntax Error: " + queryString);
            console.error(e.toString());
            return;
        }

        db.saveLog(jsonNew);

    } else if (collection == "client") {

        var jsonNew = null;

        try {
            jsonNew = JSON.parse(query[1]);
        } catch (e) {
            console.log("Syntax Error: " + query[1]);
            console.error(e.toString());
            return;
        }

        db.saveClient(jsonNew);

    } else if (collection == "complete") {

        var jsonNew = null;

        try {
            jsonNew = JSON.parse(query[1]);
        } catch (e) {
            console.log("Syntax Error: " + query[1]);
            console.error(e.toString());
            return;
        }

        console.log(query[1]);

        db.saveComplete(jsonNew);

    } else if (collection == "error") {

        var jsonNew = null;

        try {
            jsonNew = JSON.parse(query[1]);
        } catch (e) {
            console.log("Syntax Error: " + query[1]);
            console.error(e.toString());
            return;
        }

        db.saveError(jsonNew);

    }

};

var nodeCallback = function(document) {
    save(document);
};

nbus.join(MGS_NODE_NAME, nodeCallback, true);