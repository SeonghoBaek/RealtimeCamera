var express = require('express');
var router = express.Router();
var fs = require('fs');
var url = require('url');
var db = require('./mgs');

router.get('/', function(req, res) {
    var query = url.parse(req.url, true).query;

    if (query.type == "all") {

        console.log("All Query");

        db.client.find({}, {"_id":0, "status":1, "client":1, "tid":1, "session":1, "date": 1}, function(err,clients) {
            res.send(clients);
        });

    } else if (query.type == "run") {

        console.log("Run Query");

        db.client.find({"status":"CLIENT_RUN"}, {"_id":0, "client":1, "tid":1, "session":1, "date": 1}, function(err,clients) {
            res.send(clients);
        });

    } else if (query.type == "wait") {

        console.log("Wait Query");

        db.client.find({"status":"CLIENT_WAIT"}, {"_id":0, "client":1}, function(err,clients) {
            res.send(clients);
        });

    } else {
        console.log("Invalid Query");
    }
});

router.post('/', function(req, res, next) {
    req.busboy.on('file', function(fieldName, file, fileName) {});

    req.busboy.on('field', function(key, value) {});

    req.pipe(req.busboy);
});

module.exports = router;











