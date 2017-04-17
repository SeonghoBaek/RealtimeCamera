var express = require('express');
var router = express.Router();
var fs = require('fs');

router.get('/', function(req, res) {
    console.log("Welcome to wilson");
    fs.readFile('wilson/wilson.html', function(err, data) {
        if (err) console.log(err);
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(data);
    });
});

router.post('/', function(req, res, next) {
    req.busboy.on('file', function(fieldName, file, fileName) {});

    req.busboy.on('field', function(key, value) {});

    req.pipe(req.busboy);
});

module.exports = router;











