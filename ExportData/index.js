require('dotenv').config();
const env = process.env;
const mysql = require('mysql');

var database;

dbInit();

function dbInit()
{
    database = mysql.createConnection({
        connectionLimit: 5,
        host: env.MYSQL_HOST,
        user: env.MYSQL_USER,
        password: env.MYSQL_PASSWORD,
        database: env.MYSQL_DB
    });

    database.connect(function(err) {
        if (err) {
            console.log("DB-Error: " + err);
            return;
        }
        console.log("Connected to database!");
        getData();
    });
}

function getData()
{
    let query = "SELECT component_path , label_path FROM images WHERE looked_at >= 3 AND validated/looked_at > 0.49 LIMIT 0,1000000;"
    database.query(query, (err, result) => {
        if (err) {
            console.error(err);
        } else if (result.length >= 1) {
            console.log(result);
        }
    });
}

function readData()
{

}

function writeData()
{

}