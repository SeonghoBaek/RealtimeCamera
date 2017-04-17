/**
 * Created by major on 9/8/14.
 */
var express = require('express');
var router = express.Router();
var fs = require('fs');
var nbus = require('./nodebus');

router.get('/', function(req, res) {
    console.log("Welcome to DRUWA");
    fs.readFile('wilson/wilson.html', function(err, data) {
        if (err) console.log(err);
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(data);
    });
});

router.post('/', function(req, res, next) {
    console.log("Request");

    //res.redirect('/');

    nbus.send("robot_bridge", "open")
    fs.readFile('wilson/confirm.html', function(err, data) {
        if (err) console.log(err);
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(data);
    });
});

module.exports = router;











