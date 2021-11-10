require('dotenv').config();
const env = process.env;
const mysql = require('mysql');
const path = require('path');
const fs = require('fs');

var database;
var count;

createFolder();
dbInit();


function createFolder()
{
    if(!fs.existsSync(env.DATAFOLDER))
    {
        fs.mkdirSync(env.DATAFOLDER);
    }
    let data = fs.readdirSync(env.GETFOLDER);
    for(var i in data)
    {
        let dir = path.join(env.DATAFOLDER, data[i], "/");
        if(!fs.existsSync(dir))
        {
            fs.mkdirSync(dir);
        }
    }
}

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
        getData();
    });
}

function getData()
{
    let query = "SELECT component_path , label_path FROM images WHERE looked_at >= 3 AND validated/looked_at > 0.51 LIMIT 0,1000000;"
    database.query(query, (err, result) => {
        if (err) {
            console.error(err);
        } else if (result.length >= 1) {
            count = 0;
            for(var i in result)
            {
                readData(result[i]);
            }
            console.log(`${count} out of ${result.length} entries copied`);
            finish();
        }
    });
}

function readData(data)
{
    let par = data.component_path.split('/');
    let dest_component = path.join(env.DATAFOLDER, par[3], par[4]);
    let loc_component = path.join(env.GETFOLDER, par[3], par[4]);

    par = data.label_path.split('/');
    let dest_label = path.join(env.DATAFOLDER, par[3], par[4]);
    let loc_label = path.join(env.GETFOLDER, par[3], par[4]);

    writeData(dest_component, loc_component, dest_label, loc_label, data)
}

function writeData(dest_component, loc_component, dest_label, loc_label, data)
{
    if( (fs.statSync(loc_component).size < 1000) || (fs.statSync(loc_label).size < 1000)) return; // Delete out of DB?
    if(fs.existsSync(dest_component)) fs.unlinkSync(dest_component);
    if(fs.existsSync(dest_label)) fs.unlinkSync(dest_label);
    fs.copyFileSync(loc_component, dest_component);
    fs.copyFileSync(loc_label, dest_label);
    count++;
}

function finish()
{
    database.end();
}